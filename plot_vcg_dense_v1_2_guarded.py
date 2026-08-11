#!/usr/bin/env python3
"""Plot the panel-B-guarded VCG-Dense V1.2 development diagnostic.

All coordinates are authenticated by ``guarded-selection-report.json`` and
its frozen source Pareto report.  Panel B informed the guarded checkpoint
choice, so the figure deliberately presents descriptive development evidence,
never confirmatory or deployment evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_vcg_dense_pareto import (
    ENHANCED_GA_COLOR,
    FINAL_COLOR,
    NEAREST_COLOR,
    SELECTED_COLOR,
    ParetoPoint,
    build_plot_spec as build_source_plot_spec,
    load_report as load_source_report,
)


GUARDED_PROTOCOL = "vcg_dense_v1_2_panel_b_guarded_selection_development_v1"
GUARDED_SCOPE = "panel_B_selected_posthoc_development_diagnostic_only"
GUARD_RULE = (
    "validation_A_relocation_rank_then_panel_B_strict_full_"
    "mae_cost_le_2_positive_relocation_saving_sequential_accept_v2"
)
EXPECTED_GUARDED_EPISODES = {0: 475, 1: 500, 2: 500}
DEFAULT_GUARDED_REPORT = Path(
    "results/vcg-dense-v1-2-posthoc-development-30seed/"
    "guarded-selection-report.json"
)
DEFAULT_SOURCE_REPORT = Path(
    "results/vcg-dense-v1-1-pareto-development-30seed/pareto-report.json"
)
DEFAULT_FIGURE_NAME = "vcg-dense-v1-2-guarded-pareto"
ENHANCED_ID = "duration_aware_enhanced_complete_rolling_ga"
NEAREST_ID = "duration_aware_nearest_free"

GUARDED_COLOR = "#7B3294"
GUARDED_LIGHT = "#E7D8EF"
SHIFT_COLOR = "#5A2670"
SEED_SHIFT_COLOR = "#9A72AC"


@dataclass(frozen=True)
class GuardedShift:
    model_seed: int
    chosen_episode: int
    selected: ParetoPoint
    guarded: ParetoPoint
    guard_mae_cost: float
    guard_relocation_saving: float


@dataclass(frozen=True)
class GuardedPlotSpec:
    selected: ParetoPoint
    guarded: ParetoPoint
    episode500: ParetoPoint
    enhanced_ga: ParetoPoint
    nearest_free: ParetoPoint
    seed_shifts: tuple[GuardedShift, ...]
    instance_count: int
    guard_mae_margin: float
    guard_rule: str
    positive_relocation_gate: bool
    mandatory_reference_fallback: bool
    guarded_mae_delta_vs_selected: float
    guarded_relocation_delta_vs_selected: float
    guarded_mae_delta_vs_enhanced: float
    guarded_relocation_delta_vs_enhanced: float


def _mapping(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_guarded_report(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("guarded report must contain a JSON object")
    return payload


def _summary_point(
    point_id: str,
    label: str,
    summary: Mapping,
    *,
    group: str,
    diagnostic: bool,
    seed: Optional[int] = None,
) -> ParetoPoint:
    if int(summary.get("episodes", 0)) != 30:
        raise ValueError(f"{point_id} must contain 30 episodes")
    if (
        float(summary.get("strict_method_success_rate", 0.0)) != 1.0
        or float(summary.get("completion_rate", 0.0)) != 1.0
    ):
        raise ValueError(f"{point_id} is not strict/full on panel B")
    return ParetoPoint(
        point_id=point_id,
        label=label,
        mae=_finite(summary.get("mean_absolute_error"), f"{point_id}.MAE"),
        relocations_per_100=_finite(
            summary.get("relocations_per_100_deliveries"),
            f"{point_id}.relocations_per_100",
        ),
        policy_group=group,
        diagnostic_only=diagnostic,
        model_seed=seed,
    )


def _assert_same_point(left: ParetoPoint, right: ParetoPoint, name: str) -> None:
    if not (
        _close(left.mae, right.mae)
        and _close(left.relocations_per_100, right.relocations_per_100)
    ):
        raise ValueError(f"{name} disagrees with the source Pareto report")


def _validate_source_provenance(
    guarded: Mapping,
    source_report_path: Optional[Path],
) -> None:
    if source_report_path is None:
        return
    source_report_path = Path(source_report_path).resolve()
    record = _mapping(
        _mapping(guarded.get("input_hashes"), "input_hashes").get(
            "source/pareto-report.json"
        ),
        "input_hashes.source/pareto-report.json",
    )
    recorded_path = Path(str(record.get("path"))).resolve()
    if recorded_path != source_report_path:
        raise ValueError("source Pareto report path disagrees with guarded provenance")
    recorded_hash = str(record.get("sha256", ""))
    if len(recorded_hash) != 64 or _sha256(source_report_path) != recorded_hash:
        raise ValueError("source Pareto report hash disagrees with guarded provenance")


def _validate_trial_sequence(
    sequential: Mapping,
    *,
    model_seed: int,
    chosen_episode: int,
    reference_episode: int,
    margin: float,
) -> tuple[float, float]:
    if sequential.get("guard_rule") != GUARD_RULE:
        raise ValueError(f"seed {model_seed} does not use the frozen v2 guard")
    if int(sequential.get("model_seed", -1)) != model_seed:
        raise ValueError(f"seed {model_seed} sequential-guard seed mismatch")
    if int(sequential.get("chosen_episode", -1)) != chosen_episode:
        raise ValueError(f"seed {model_seed} sequential choice mismatch")
    if int(sequential.get("reference_episode", -1)) != reference_episode:
        raise ValueError(f"seed {model_seed} reference episode mismatch")

    rank = sequential.get("validation_A_relocation_rank")
    trials = sequential.get("sequential_panel_B_trials")
    if not isinstance(rank, list) or not rank:
        raise ValueError(f"seed {model_seed} has no frozen validation-A rank")
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"seed {model_seed} has no panel-B guard trials")

    accepted = []
    for trial_index, item in enumerate(trials, start=1):
        trial = _mapping(item, f"seed {model_seed} trial {trial_index}")
        is_fallback = trial.get("selected_as_mandatory_reference_fallback") is True
        if is_fallback:
            if trial_index != len(trials):
                raise ValueError("mandatory reference fallback must be the last trial")
            if int(trial.get("checkpoint_episode", -1)) != reference_episode:
                raise ValueError("mandatory fallback does not select the reference")
            if trial.get("accepted") is not False:
                raise ValueError("mandatory reference fallback cannot be a passing candidate")
            continue

        if int(trial.get("validation_A_rank", -1)) != trial_index:
            raise ValueError(f"seed {model_seed} panel-B trials are not sequential")
        ranked = _mapping(rank[trial_index - 1], "validation-A rank entry")
        if int(ranked.get("checkpoint_episode", -1)) != int(
            trial.get("checkpoint_episode", -2)
        ):
            raise ValueError(f"seed {model_seed} trial order differs from validation A")
        if trial.get("available_authenticated_evaluation") is not True:
            if trial.get("accepted") is not False:
                raise ValueError("an unavailable evaluation cannot pass the guard")
            continue

        if trial.get("panel_B_positive_relocation_saving_required") is not True:
            raise ValueError("positive panel-B relocation saving is not mandatory")
        strict_full = trial.get("strict_full_panel_B") is True
        mae_cost = _finite(trial.get("panel_B_mae_cost"), "panel-B MAE cost")
        trial_margin = _finite(
            trial.get("panel_B_mae_margin"), "panel-B MAE margin"
        )
        relocation_saving = _finite(
            trial.get("panel_B_relocation_saving_per_100_deliveries"),
            "panel-B relocation saving",
        )
        if not _close(trial_margin, margin):
            raise ValueError("panel-B trial uses the wrong MAE margin")
        should_accept = strict_full and mae_cost <= margin and relocation_saving > 0.0
        if trial.get("accepted") is not should_accept:
            raise ValueError("panel-B accepted flag disagrees with the frozen guard")
        if should_accept:
            if trial.get("reason") != "first_available_ranked_candidate_passing_guard":
                raise ValueError("accepted panel-B trial has the wrong reason")
            accepted.append((trial_index, int(trial["checkpoint_episode"]), mae_cost, relocation_saving))

    if len(accepted) != 1:
        raise ValueError(f"seed {model_seed} must have exactly one passing choice")
    trial_index, episode, mae_cost, relocation_saving = accepted[0]
    if trial_index != len(trials) or episode != chosen_episode:
        raise ValueError("guard did not stop at its first passing candidate")
    return mae_cost, relocation_saving


def _validate_metric(
    metrics: Mapping,
    metric_name: str,
    expected_mean: float,
    expected_per_seed: Mapping[int, float],
) -> float:
    metric = _mapping(metrics.get(metric_name), metric_name)
    mean = _finite(metric.get("mean_across_training_seeds"), f"{metric_name}.mean")
    if not _close(mean, expected_mean):
        raise ValueError(f"{metric_name} aggregate contrast is inconsistent")
    if int(metric.get("n_training_seeds", 0)) != 3:
        raise ValueError(f"{metric_name} must use three independent training seeds")
    if int(metric.get("n_paired_instances_per_seed", 0)) != 30:
        raise ValueError(f"{metric_name} must use 30 paired instances per seed")
    rows = metric.get("per_training_seed")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError(f"{metric_name} per-seed contrasts are incomplete")
    observed = {}
    for row in rows:
        row = _mapping(row, f"{metric_name}.per_training_seed")
        seed = int(row.get("model_seed", -1))
        observed[seed] = _finite(row.get("panel_mean_delta"), f"{metric_name}.seed{seed}")
    if set(observed) != {0, 1, 2}:
        raise ValueError(f"{metric_name} training seeds are incomplete")
    for seed, expected in expected_per_seed.items():
        if not _close(observed[seed], expected):
            raise ValueError(f"{metric_name} seed {seed} contrast is inconsistent")
    return mean


def _validate_contrast(
    contrasts: Mapping,
    name: str,
    comparator_name: str,
    guarded: ParetoPoint,
    comparator: ParetoPoint,
    shifts: Sequence[GuardedShift],
) -> tuple[float, float]:
    contrast = _mapping(contrasts.get(name), name)
    if contrast.get("comparator") != comparator_name:
        raise ValueError(f"{name} comparator is inconsistent")
    metrics = _mapping(contrast.get("metrics"), f"{name}.metrics")
    conventions = _mapping(contrast.get("sign_convention"), f"{name}.signs")
    if conventions.get("mae_cost") != "negative favors posthoc":
        raise ValueError(f"{name} MAE sign convention is inconsistent")
    if (
        conventions.get("relocations_saving_per_100_deliveries")
        != "positive favors posthoc"
    ):
        raise ValueError(f"{name} relocation sign convention is inconsistent")

    if comparator_name == "selected_best":
        per_seed_mae = {
            shift.model_seed: shift.guarded.mae - shift.selected.mae
            for shift in shifts
        }
        per_seed_saving = {
            shift.model_seed: (
                shift.selected.relocations_per_100
                - shift.guarded.relocations_per_100
            )
            for shift in shifts
        }
    else:
        per_seed_mae = {
            shift.model_seed: shift.guarded.mae - comparator.mae
            for shift in shifts
        }
        per_seed_saving = {
            shift.model_seed: (
                comparator.relocations_per_100
                - shift.guarded.relocations_per_100
            )
            for shift in shifts
        }

    mae_delta = _validate_metric(
        metrics,
        "mae_cost",
        guarded.mae - comparator.mae,
        per_seed_mae,
    )
    relocation_saving = _validate_metric(
        metrics,
        "relocations_saving_per_100_deliveries",
        comparator.relocations_per_100 - guarded.relocations_per_100,
        per_seed_saving,
    )
    return mae_delta, -relocation_saving


def build_guarded_plot_spec(
    guarded_report: Mapping,
    source_report: Mapping,
    *,
    source_report_path: Optional[Path] = None,
) -> GuardedPlotSpec:
    """Validate the guarded report and extract descriptive plot coordinates."""

    if guarded_report.get("protocol") != GUARDED_PROTOCOL:
        raise ValueError(f"unexpected guarded protocol: {guarded_report.get('protocol')!r}")
    if guarded_report.get("scope") != GUARDED_SCOPE:
        raise ValueError("guarded scope is not panel-B-selected development-only")
    if guarded_report.get("guard_rule") != GUARD_RULE:
        raise ValueError("guarded report does not use the frozen v2 selection rule")
    margin = _finite(guarded_report.get("guard_mae_margin"), "guard MAE margin")
    if not _close(margin, 2.0):
        raise ValueError("guard MAE margin must be 2.0")
    for false_field in (
        "performance_claim_authorized",
        "deployment_checkpoint_eligible",
        "sealed_panels_opened",
    ):
        if guarded_report.get(false_field) is not False:
            raise ValueError(f"guarded report has unsafe status: {false_field}")
    for zero_field in ("new_policy_executions", "new_baseline_executions"):
        if int(guarded_report.get(zero_field, -1)) != 0:
            raise ValueError(f"guarded report unexpectedly executed: {zero_field}")

    guardrails = _mapping(guarded_report.get("guardrails"), "guardrails")
    required_true = (
        "candidate_rank_frozen_on_validation_A",
        "guard_panel_used_for_selection",
        "panel_B_selected",
        "positive_panel_B_relocation_saving_required",
        "reference_is_mandatory_last_fallback",
        "not_an_unbiased_test_panel",
        "no_rollout_code_path",
        "no_claim",
    )
    for field in required_true:
        if guardrails.get(field) is not True:
            raise ValueError(f"guarded report guardrail failed: {field}")
    if guardrails.get("guard_panel") != "already_opened_development_80000_80029":
        raise ValueError("guarded report uses the wrong development panel")
    _validate_source_provenance(guarded_report, source_report_path)

    source_spec = build_source_plot_spec(source_report)
    source_points = {point.point_id: point for point in source_spec.aggregate_points}
    selected = source_spec.selected_aggregate
    episode500 = source_spec.final_aggregate
    enhanced = source_points[ENHANCED_ID]
    nearest = source_points[NEAREST_ID]
    selected_by_seed = {
        shift.model_seed: shift.selected for shift in source_spec.seed_shifts
    }

    rows = guarded_report.get("per_training_seed")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("guarded report must contain exactly three training seeds")
    shifts = []
    guarded_points = []
    for row_value in sorted(rows, key=lambda item: int(item["model_seed"])):
        row = _mapping(row_value, "per_training_seed row")
        seed = int(row.get("model_seed", -1))
        if seed not in EXPECTED_GUARDED_EPISODES:
            raise ValueError(f"unexpected guarded model seed: {seed}")
        chosen_episode = int(row.get("guarded_episode", -1))
        if chosen_episode != EXPECTED_GUARDED_EPISODES[seed]:
            raise ValueError(f"guarded checkpoint choice drifted for seed {seed}")
        reference_episode = int(row.get("reference_episode", -1))

        reference = _summary_point(
            f"selected_seed{seed}",
            f"Selected s{seed}",
            _mapping(row.get("reference_summary"), "reference_summary"),
            group="selected_best",
            diagnostic=False,
            seed=seed,
        )
        _assert_same_point(reference, selected_by_seed[seed], f"selected seed {seed}")
        guarded = _summary_point(
            f"guarded_seed{seed}_ep{chosen_episode}",
            f"Guarded s{seed}",
            _mapping(row.get("guarded_summary"), "guarded_summary"),
            group="panel_b_guarded_posthoc_v1_2_diagnostic",
            diagnostic=True,
            seed=seed,
        )
        enhanced_row = _summary_point(
            f"enhanced_seed{seed}",
            "Enhanced rolling GA",
            _mapping(row.get("enhanced_ga_summary"), "enhanced_ga_summary"),
            group="baseline",
            diagnostic=False,
        )
        _assert_same_point(enhanced_row, enhanced, f"enhanced GA seed {seed}")

        sequential = _mapping(row.get("sequential_guard"), "sequential_guard")
        reference_panel = _summary_point(
            f"guard_reference_seed{seed}",
            "Guard reference",
            _mapping(
                sequential.get("reference_summary_panel_B"),
                "reference_summary_panel_B",
            ),
            group="selected_best",
            diagnostic=False,
            seed=seed,
        )
        chosen_panel = _summary_point(
            f"guard_choice_seed{seed}",
            "Guard choice",
            _mapping(
                sequential.get("chosen_summary_panel_B"),
                "chosen_summary_panel_B",
            ),
            group="panel_b_guarded_posthoc_v1_2_diagnostic",
            diagnostic=True,
            seed=seed,
        )
        _assert_same_point(reference_panel, reference, f"guard reference seed {seed}")
        _assert_same_point(chosen_panel, guarded, f"guard choice seed {seed}")
        mae_cost, relocation_saving = _validate_trial_sequence(
            sequential,
            model_seed=seed,
            chosen_episode=chosen_episode,
            reference_episode=reference_episode,
            margin=margin,
        )
        if not _close(guarded.mae - reference.mae, mae_cost):
            raise ValueError(f"seed {seed} guard MAE cost is inconsistent")
        if not _close(
            reference.relocations_per_100 - guarded.relocations_per_100,
            relocation_saving,
        ):
            raise ValueError(f"seed {seed} guard relocation saving is inconsistent")

        provenance = _mapping(
            _mapping(
                _mapping(guarded_report.get("candidate_provenance"), "provenance").get(
                    str(seed)
                ),
                f"provenance seed {seed}",
            ).get(str(chosen_episode)),
            f"provenance seed {seed} episode {chosen_episode}",
        )
        for digest_name in ("checkpoint_sha256", "deployment_policy_digest"):
            if len(str(provenance.get(digest_name, ""))) != 64:
                raise ValueError(f"seed {seed} has invalid {digest_name} provenance")

        guarded_points.append(guarded)
        shifts.append(
            GuardedShift(
                model_seed=seed,
                chosen_episode=chosen_episode,
                selected=reference,
                guarded=guarded,
                guard_mae_cost=mae_cost,
                guard_relocation_saving=relocation_saving,
            )
        )

    if {shift.model_seed for shift in shifts} != {0, 1, 2}:
        raise ValueError("guarded report training seeds are duplicated or incomplete")
    aggregate = _mapping(
        guarded_report.get("aggregate_metric_summaries"),
        "aggregate_metric_summaries",
    )
    mae_summary = _mapping(aggregate.get("mean_absolute_error"), "aggregate MAE")
    relocation_summary = _mapping(
        aggregate.get("relocations_per_100_deliveries"),
        "aggregate relocations",
    )
    guarded = ParetoPoint(
        point_id="panel_b_guarded_v1_2",
        label="Guarded V1.2 (475/500/500)",
        mae=_finite(mae_summary.get("mean"), "guarded aggregate MAE"),
        relocations_per_100=_finite(
            relocation_summary.get("mean"), "guarded aggregate relocations"
        ),
        policy_group="panel_b_guarded_posthoc_v1_2_diagnostic",
        diagnostic_only=True,
    )
    mean_mae = sum(point.mae for point in guarded_points) / 3.0
    mean_relocations = sum(point.relocations_per_100 for point in guarded_points) / 3.0
    if not _close(guarded.mae, mean_mae) or not _close(
        guarded.relocations_per_100, mean_relocations
    ):
        raise ValueError("guarded aggregate is not equal-seed weighted")
    if int(mae_summary.get("degrees_of_freedom", -1)) != 2 or int(
        relocation_summary.get("degrees_of_freedom", -1)
    ) != 2:
        raise ValueError("guarded aggregate must treat training seed as n=3")

    contrasts = _mapping(guarded_report.get("aggregate_contrasts"), "contrasts")
    selected_mae_delta, selected_relocation_delta = _validate_contrast(
        contrasts,
        "guarded_vs_selected_best",
        "selected_best",
        guarded,
        selected,
        shifts,
    )
    enhanced_mae_delta, enhanced_relocation_delta = _validate_contrast(
        contrasts,
        "guarded_vs_enhanced_ga",
        "enhanced_ga",
        guarded,
        enhanced,
        shifts,
    )
    if not (selected_mae_delta > 0.0 and selected_relocation_delta < 0.0):
        raise ValueError("guarded/selected point relation is not the reported tradeoff")
    if not (enhanced_mae_delta < 0.0 and enhanced_relocation_delta > 0.0):
        raise ValueError("guarded/enhanced point relation is not the reported tradeoff")

    return GuardedPlotSpec(
        selected=selected,
        guarded=guarded,
        episode500=episode500,
        enhanced_ga=enhanced,
        nearest_free=nearest,
        seed_shifts=tuple(shifts),
        instance_count=30,
        guard_mae_margin=margin,
        guard_rule=GUARD_RULE,
        positive_relocation_gate=True,
        mandatory_reference_fallback=True,
        guarded_mae_delta_vs_selected=selected_mae_delta,
        guarded_relocation_delta_vs_selected=selected_relocation_delta,
        guarded_mae_delta_vs_enhanced=enhanced_mae_delta,
        guarded_relocation_delta_vs_enhanced=enhanced_relocation_delta,
    )


def _xy(point: ParetoPoint) -> tuple[float, float]:
    return point.mae, point.relocations_per_100


def _draw_seed_shift(ax, shift: GuardedShift, *, labels: bool) -> None:
    start, end = _xy(shift.selected), _xy(shift.guarded)
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": SEED_SHIFT_COLOR,
            "linestyle": (0, (3, 2)),
            "linewidth": 1.25,
            "mutation_scale": 9,
        },
        zorder=2,
    )
    ax.scatter(
        *start,
        s=38,
        marker="o",
        facecolor=SELECTED_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.scatter(
        *end,
        s=58,
        marker="H",
        facecolor=GUARDED_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    if labels:
        offsets = {0: (13, 10), 1: (8, 7), 2: (8, -3)}
        ax.annotate(
            f"s{shift.model_seed} → ep{shift.chosen_episode}",
            end,
            xytext=offsets[shift.model_seed],
            textcoords="offset points",
            fontsize=7.3,
            color=GUARDED_COLOR,
            fontweight="bold",
            zorder=8,
        )


def _draw_aggregate_shift(ax, spec: GuardedPlotSpec, linewidth: float) -> None:
    ax.annotate(
        "",
        xy=_xy(spec.guarded),
        xytext=_xy(spec.selected),
        arrowprops={
            "arrowstyle": "-|>",
            "color": SHIFT_COLOR,
            "linewidth": linewidth,
            "mutation_scale": 13,
            "shrinkA": 7,
            "shrinkB": 8,
        },
        zorder=6,
    )


def _draw_aggregate_points(ax, spec: GuardedPlotSpec, *, annotate: bool) -> None:
    styles = {
        "selected": (spec.selected, "*", SELECTED_COLOR, "black", 220, 0.8),
        "guarded": (spec.guarded, "H", GUARDED_COLOR, "black", 155, 0.8),
        "episode500": (spec.episode500, "D", "white", FINAL_COLOR, 118, 2.0),
        "enhanced": (spec.enhanced_ga, "s", ENHANCED_GA_COLOR, "black", 102, 0.7),
        "nearest": (spec.nearest_free, "X", NEAREST_COLOR, "black", 102, 0.7),
    }
    for point, marker, face, edge, size, width in styles.values():
        ax.scatter(
            *_xy(point),
            marker=marker,
            s=size,
            facecolor=face,
            edgecolor=edge,
            linewidth=width,
            zorder=9,
        )
    if not annotate:
        return

    labels = {
        "selected": ("V1.1 selected", -5, 16, "left", "#202020"),
        "guarded": (
            "Guarded V1.2\n(ep475/500/500)",
            48,
            -12,
            "left",
            GUARDED_COLOR,
        ),
        "episode500": (
            "Episode-500 diagnostic",
            34,
            26,
            "left",
            FINAL_COLOR,
        ),
        "enhanced": ("Enhanced rolling GA", 11, 13, "left", "#202020"),
        "nearest": ("Nearest-free", -8, 15, "right", "#202020"),
    }
    for key, (label, dx, dy, align, color) in labels.items():
        point = styles[key][0]
        arrow = None
        if key in ("guarded", "episode500"):
            arrow = {
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.8,
                "shrinkA": 2,
                "shrinkB": 5,
            }
        ax.annotate(
            label,
            _xy(point),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=8.35,
            color=color,
            fontweight="bold" if key in ("selected", "guarded", "episode500") else "normal",
            arrowprops=arrow,
            zorder=10,
        )


def _draw_inset(ax, spec: GuardedPlotSpec) -> None:
    inset = ax.inset_axes([0.505, 0.395, 0.465, 0.52])
    for shift in spec.seed_shifts:
        _draw_seed_shift(inset, shift, labels=True)
    _draw_aggregate_shift(inset, spec, 2.1)
    _draw_aggregate_points(inset, spec, annotate=False)

    detail = [spec.selected, spec.guarded, spec.episode500, spec.enhanced_ga]
    detail.extend(
        point
        for shift in spec.seed_shifts
        for point in (shift.selected, shift.guarded)
    )
    x_values = [point.mae for point in detail]
    y_values = [point.relocations_per_100 for point in detail]
    inset.set_xlim(min(x_values) - 0.5, max(x_values) + 0.65)
    inset.set_ylim(max(0, min(y_values) - 2.0), max(y_values) + 2.4)
    inset.set_title(
        "V1.1 selected → panel-B-guarded choice by seed",
        loc="left",
        fontsize=8.25,
        fontweight="bold",
        pad=5,
    )
    inset.set_xlabel("MAE", fontsize=7.5)
    inset.set_ylabel("Relocations / 100", fontsize=7.5)
    inset.tick_params(axis="both", labelsize=7)
    inset.grid(True, linestyle=":", linewidth=0.7, alpha=0.45)
    for spine in inset.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.8)


def _signed(value: float) -> str:
    return f"+{value:.3f}" if value >= 0.0 else f"−{abs(value):.3f}"


def render_guarded_plot(
    spec: GuardedPlotSpec,
    output_prefix: Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(9.35, 6.45))
    fig.subplots_adjust(left=0.105, right=0.98, top=0.88, bottom=0.235)

    for shift in spec.seed_shifts:
        _draw_seed_shift(ax, shift, labels=False)
    _draw_aggregate_shift(ax, spec, 2.3)
    _draw_aggregate_points(ax, spec, annotate=True)

    points = [
        spec.selected,
        spec.guarded,
        spec.episode500,
        spec.enhanced_ga,
        spec.nearest_free,
    ]
    points.extend(
        point
        for shift in spec.seed_shifts
        for point in (shift.selected, shift.guarded)
    )
    x_values = [point.mae for point in points]
    y_values = [point.relocations_per_100 for point in points]
    ax.set_xlim(max(0, min(x_values) - 2.5), max(x_values) + 3.0)
    ax.set_ylim(-1.8, max(y_values) + 3.2)
    ax.set_xlabel("Mean absolute timing error (steps)")
    ax.set_ylabel("Relocations per 100 deliveries")
    ax.set_title(
        "Guarded V1.2 timing–relocation diagnostic",
        loc="left",
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.0,
        1.012,
        f"{spec.instance_count} paired panel-B instances · equal-weight 3-seed aggregates",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#4A4A4A",
    )
    ax.text(
        0.995,
        1.012,
        "DEVELOPMENT ONLY · PANEL-B SELECTED · NOT DEPLOYMENT ELIGIBLE",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.15,
        color="#5A2670",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#F8F0FB",
            "edgecolor": GUARDED_LIGHT,
            "linewidth": 0.8,
        },
    )
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    ax.annotate(
        "Lower-left is better",
        xy=(0.035, 0.055),
        xytext=(0.20, 0.19),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=8,
        color="#4A4A4A",
        arrowprops={"arrowstyle": "-|>", "color": "#777777", "lw": 1.0},
    )
    _draw_inset(ax, spec)

    handles = [
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=12,
            markerfacecolor=SELECTED_COLOR,
            markeredgecolor="black",
            label="V1.1 selected aggregate",
        ),
        Line2D(
            [],
            [],
            marker="H",
            linestyle="none",
            markersize=8,
            markerfacecolor=GUARDED_COLOR,
            markeredgecolor="black",
            label="Guarded V1.2 (panel-B selected)",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=FINAL_COLOR,
            markeredgewidth=1.6,
            label="Episode-500 aggregate (diagnostic)",
        ),
        Line2D(
            [],
            [],
            color=SEED_SHIFT_COLOR,
            linestyle=(0, (3, 2)),
            marker="o",
            markersize=4,
            markerfacecolor=SELECTED_COLOR,
            label="Per-seed selected → guarded",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.70, 0.105),
        frameon=True,
        framealpha=0.96,
        edgecolor="#CCCCCC",
    )

    fig.text(
        0.105,
        0.105,
        (
            "Descriptive panel-B aggregate shifts (guarded − comparator):  "
            f"vs V1.1 selected  ΔMAE {_signed(spec.guarded_mae_delta_vs_selected)}, "
            f"Δrelocations {_signed(spec.guarded_relocation_delta_vs_selected)}/100   |   "
            f"vs enhanced GA  ΔMAE {_signed(spec.guarded_mae_delta_vs_enhanced)}, "
            f"Δrelocations {_signed(spec.guarded_relocation_delta_vs_enhanced)}/100"
        ),
        ha="left",
        va="center",
        fontsize=8.15,
        color="#542364",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#F8F0FB",
            "edgecolor": GUARDED_LIGHT,
            "linewidth": 0.8,
        },
    )
    fig.text(
        0.105,
        0.047,
        (
            "Guard: strict/full + MAE cost ≤2 + positive relocation saving; "
            "selected V1.1 is the mandatory fallback. Panel B informed selection—"
            "no inferential or deployment claim."
        ),
        ha="left",
        va="center",
        fontsize=7.65,
        color="#555555",
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        metadata={
            "Title": "VCG-Dense V1.2 panel-B-guarded timing-relocation diagnostic",
            "Description": "Development-only, panel-B-selected diagnostic",
        },
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
        bbox_inches="tight",
        metadata={
            "Title": "VCG-Dense V1.2 panel-B-guarded timing-relocation diagnostic",
            "Subject": "Development-only panel-B-selected diagnostic",
            "Keywords": "VCG Dense, guarded selection, panel B, development diagnostic",
        },
    )
    plt.close(fig)
    return png_path, pdf_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the frozen VCG-Dense V1.2 guarded-selection report"
    )
    parser.add_argument(
        "--guarded-report", type=Path, default=DEFAULT_GUARDED_REPORT
    )
    parser.add_argument(
        "--source-pareto-report", type=Path, default=DEFAULT_SOURCE_REPORT
    )
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> tuple[Path, Path]:
    args = build_parser().parse_args(argv)
    guarded = load_guarded_report(args.guarded_report)
    source = load_source_report(args.source_pareto_report)
    spec = build_guarded_plot_spec(
        guarded,
        source,
        source_report_path=args.source_pareto_report,
    )
    prefix = args.output_prefix
    if prefix is None:
        prefix = args.guarded_report.parent / DEFAULT_FIGURE_NAME
    paths = render_guarded_plot(spec, prefix, dpi=args.dpi)
    print(f"PNG: {paths[0]}")
    print(f"PDF: {paths[1]}")
    return paths


if __name__ == "__main__":
    main()
