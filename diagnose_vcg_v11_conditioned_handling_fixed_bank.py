#!/usr/bin/env python3
"""Decompose conditioned VCG merits on one fixed bank of training-probe states.

This is a post-training diagnostic, not an evaluation.  A small common bank is
collected from the finalized seed-0 policy on the predeclared training probes.
Every terminal model is then queried on exactly the same states, candidates,
and lambda coordinates.  No simulator outcome is used to select a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Mapping, Optional, Sequence

import torch

import benchmark_viability_critic_priority as benchmark
import run_vcg_v11_conditioned_handling_damped_evaluation_85k as seed0_eval
import run_vcg_v11_conditioned_handling_seed0_85k as seed0_parent
import run_vcg_v11_conditioned_handling_two_phase_seeds12 as seeds12
import run_vcg_v11_nested_handling_pilot as pilot
import train_vcg_v11_conditioned_handling_iterative as training
from vcg_v11_nested_handling import detached_v11_features
from viability_graph_hierarchy import ID_TO_MODE
from viability_graph_preference_conditioned import select_hierarchical_index


PROTOCOL = "vcg_v1_1_conditioned_handling_fixed_merit_bank_v1"
LAMBDA_GRID = tuple(index / 80.0 for index in range(17))
MAX_FRONTIERS_PER_PROBE = 4
OUTPUT_RELATIVE = Path(
    "results/vcg-v1-1-conditioned-handling-fixed-merit-bank"
)
SEEDS12_RELATIVE = Path(
    "results/vcg-v1-1-conditioned-handling-seeds12-two-phase-development"
)
REPORT_NAME = "fixed-merit-bank-report.json"
VALUES_NAME = "fixed-merit-bank-values.csv"
SELECTIONS_NAME = "fixed-merit-bank-selections.csv"


class FixedBankDiagnosticError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FixedBankDiagnosticError(f"missing required artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping) -> None:
    pilot._atomic_json(path, value)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class _CaptureAgent:
    def __init__(self, agent, *, limit: int) -> None:
        self.agent = agent
        self.config = agent.config
        self.limit = int(limit)
        self.frontiers = []

    def reset_episode_state(self):
        return self.agent.reset_episode_state()

    def select(
        self,
        snapshot,
        *,
        training=False,
        epsilon=0.0,
        preference_lambda=None,
    ):
        kwargs = {"training": training, "epsilon": epsilon}
        if preference_lambda is not None:
            kwargs["preference_lambda"] = preference_lambda
        decision = self.agent.select(snapshot, **kwargs)
        if len(self.frontiers) < self.limit:
            prepared = decision.prepared_snapshot
            self.frontiers.append(
                {
                    "decision_epoch": int(prepared.decision_epoch),
                    "records": tuple(prepared.records),
                }
            )
        return decision

    def observe_outcome(self, decision, *, next_snapshot, done):
        return self.agent.observe_outcome(
            decision, next_snapshot=next_snapshot, done=done
        )


def _frontier_digest(records) -> str:
    semantic = tuple(
        (
            item.key,
            int(item.mode_id),
            item.action_type,
            repr(item.current_state),
            repr(item.successor_state),
            tuple(float(value) for value in item.action_features),
        )
        for item in records
    )
    return hashlib.sha256(repr(semantic).encode("utf-8")).hexdigest()


def _authenticate(project_root: Path, *, max_steps: int):
    seed0 = seed0_eval._damped_artifacts(project_root)
    output_root = project_root / SEEDS12_RELATIVE
    contract = seeds12._require_contract(
        project_root, output_root, max_steps=max_steps
    )
    inputs = seeds12._authenticate_inputs(project_root)
    report_path = output_root / seeds12.REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    terminals = {}
    summaries = {}
    for seed in seeds12.MODEL_SEEDS:
        paths = seeds12._seed_paths(output_root, seed)
        terminal = torch.load(
            paths["terminal"], map_location="cpu", weights_only=False
        )
        seeds12._validate_checkpoint(
            terminal, contract, seed=seed, terminal=True
        )
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        if terminal.get("stability_gate") != summary.get("stability_gate"):
            raise FixedBankDiagnosticError(
                f"seed-{seed} terminal stability record changed"
            )
        terminals[seed] = terminal
        summaries[seed] = summary
    return {
        "seed0": seed0,
        "seeds12_root": output_root,
        "seeds12_contract": contract,
        "seeds12_inputs": inputs,
        "seeds12_report": report,
        "seeds12_report_sha256": _sha256(report_path),
        "terminals": terminals,
        "summaries": summaries,
    }


def _collect_bank(project_root: Path, authenticated, *, device: torch.device):
    arm = authenticated["seeds12_inputs"]["arms"][0]
    trained = authenticated["seed0"]
    bank_by_digest = {}
    probe_runs = []
    for probe_index, behavior_lambda in enumerate(training.PROBE_LAMBDAS):
        instance_seed = training.PROBE_SEED_BASE + probe_index
        instance = benchmark._make_env(arm.payload).sample_episode_instance(
            instance_seed
        )
        holder = []

        def factory(base):
            agent = seed0_parent._new_terminal_agent(
                project_root,
                arm,
                trained["terminal"],
                device,
                base=base,
            )
            agent.set_epsilon(0.0)
            captured = _CaptureAgent(agent, limit=MAX_FRONTIERS_PER_PROBE)
            holder.append(captured)
            return seed0_parent._FixedLambdaAgent(captured, behavior_lambda)

        raw = pilot._run_raw(
            arm, instance, device=device, wrapper_factory=factory
        )
        if not raw.get("strict_method_success"):
            raise FixedBankDiagnosticError(
                f"training probe {instance_seed} did not complete strictly: "
                f"{raw.get('method_failure_reason')}"
            )
        if len(holder) != 1:
            raise FixedBankDiagnosticError("capture wrapper was not constructed once")
        probe_runs.append(
            {
                "probe_index": probe_index,
                "instance_seed": instance_seed,
                "behavior_lambda": float(behavior_lambda),
                "episode_instance_id": instance.instance_id,
                "captured_frontiers": len(holder[0].frontiers),
                "macro_decisions": int(raw["macro_decisions"]),
            }
        )
        for local_index, frontier in enumerate(holder[0].frontiers):
            records = frontier["records"]
            digest = _frontier_digest(records)
            source = {
                "probe_index": probe_index,
                "instance_seed": instance_seed,
                "behavior_lambda": float(behavior_lambda),
                "decision_index": local_index,
                "decision_epoch": frontier["decision_epoch"],
            }
            if digest in bank_by_digest:
                bank_by_digest[digest]["sources"].append(source)
            else:
                bank_by_digest[digest] = {
                    "digest": digest,
                    "records": records,
                    "sources": [source],
                }
    bank = []
    for index, item in enumerate(bank_by_digest.values()):
        bank.append({**item, "state_id": f"state-{index:03d}"})
    if not bank:
        raise FixedBankDiagnosticError("fixed merit bank is empty")
    return bank, probe_runs


def _load_agents(project_root: Path, authenticated, *, device: torch.device):
    inputs = authenticated["seeds12_inputs"]
    seed0_arm = inputs["arms"][0]
    agents = {
        0: seed0_parent._new_terminal_agent(
            project_root,
            seed0_arm,
            authenticated["seed0"]["terminal"],
            device,
        )
    }
    for seed in seeds12.MODEL_SEEDS:
        agents[seed] = seeds12._load_agent(
            authenticated["terminals"][seed],
            inputs,
            seed=seed,
            device=device,
        )
    for agent in agents.values():
        agent.set_epsilon(0.0)
    return agents


def _runner_up_margin(values, indices, selected_index: int) -> Optional[float]:
    alternatives = [float(values[index]) for index in indices if index != selected_index]
    if not alternatives:
        return None
    return float(values[selected_index]) - max(alternatives)


def _decompose(agent, bank):
    value_rows = []
    selection_rows = []
    candidate_sequences = {}
    state_winners = {}
    for state in bank:
        records = state["records"]
        features = detached_v11_features(agent.base_agent.Q_local, records)
        with torch.no_grad():
            qop = agent.base_agent.Q_local.q_head(features).squeeze(-1)
        immediate = torch.as_tensor(
            [1.0 if item.action_type == "reconfigure" else 0.0 for item in records],
            dtype=features.dtype,
            device=features.device,
        )
        keys = tuple(item.key for item in records)
        modes = tuple(int(item.mode_id) for item in records)
        state_winners[state["state_id"]] = []
        for value in LAMBDA_GRID:
            with torch.no_grad():
                future = agent.handling_network(features, value)
                total = immediate + future
                penalty = value * total
                merit = qop - penalty
            hierarchy = select_hierarchical_index(
                merit,
                modes,
                agent.base_agent.within_temperatures,
                candidate_keys=keys,
            )
            selected_index = hierarchy.selected_index
            selected_mode = hierarchy.selected_mode_id
            same_mode = [
                index for index, mode in enumerate(modes) if mode == selected_mode
            ]
            candidate_margin = _runner_up_margin(
                merit.detach().cpu().tolist(), same_mode, selected_index
            )
            mode_values = dict(hierarchy.mode_values)
            other_modes = [
                score for mode, score in hierarchy.mode_values if mode != selected_mode
            ]
            mode_margin = (
                None
                if not other_modes
                else float(mode_values[selected_mode] - max(other_modes))
            )
            winner = keys[selected_index]
            state_winners[state["state_id"]].append(winner)
            selection_rows.append(
                {
                    "seed": agent.seed,
                    "state_id": state["state_id"],
                    "lambda": value,
                    "selected_candidate_index": selected_index,
                    "selected_key": winner,
                    "selected_action_type": records[selected_index].action_type,
                    "selected_mode": ID_TO_MODE[selected_mode],
                    "selected_qop": float(qop[selected_index]),
                    "selected_qn": float(total[selected_index]),
                    "selected_lambda_qn": float(penalty[selected_index]),
                    "selected_merit": float(merit[selected_index]),
                    "within_mode_margin": candidate_margin,
                    "mode_margin": mode_margin,
                }
            )
            for index, record in enumerate(records):
                identifier = (state["state_id"], index)
                candidate_sequences.setdefault(identifier, {"qn": [], "penalty": []})
                candidate_sequences[identifier]["qn"].append(float(total[index]))
                candidate_sequences[identifier]["penalty"].append(
                    float(penalty[index])
                )
                value_rows.append(
                    {
                        "seed": agent.seed,
                        "state_id": state["state_id"],
                        "candidate_index": index,
                        "candidate_key": record.key,
                        "action_type": record.action_type,
                        "mode": ID_TO_MODE[record.mode_id],
                        "lambda": value,
                        "qop": float(qop[index]),
                        "qn_immediate": int(immediate[index].item()),
                        "qn_future": float(future[index]),
                        "qn_total": float(total[index]),
                        "lambda_qn": float(penalty[index]),
                        "merit": float(merit[index]),
                        "selected": index == selected_index,
                    }
                )
    return value_rows, selection_rows, candidate_sequences, state_winners


def _direction_reversals(values: Sequence[float], *, tolerance: float = 1e-6) -> int:
    signs = []
    for left, right in zip(values, values[1:]):
        difference = float(right) - float(left)
        if abs(difference) > tolerance:
            signs.append(1 if difference > 0 else -1)
    return sum(left != right for left, right in zip(signs, signs[1:]))


def summarize_sequences(candidate_sequences, state_winners) -> dict:
    diagnostics = []
    for sequence in candidate_sequences.values():
        qn = sequence["qn"]
        penalty = sequence["penalty"]
        differences = [right - left for left, right in zip(qn, qn[1:])]
        penalty_differences = [
            right - left for left, right in zip(penalty, penalty[1:])
        ]
        variation = sum(abs(value) for value in differences)
        displacement = abs(qn[-1] - qn[0])
        diagnostics.append(
            {
                "endpoint_decreased": qn[-1] < qn[0] - 1e-6,
                "nonincreasing": all(value <= 1e-6 for value in differences),
                "direction_reversals": _direction_reversals(qn),
                "variation": variation,
                "excess_variation_fraction": (
                    0.0 if variation <= 1e-12 else (variation - displacement) / variation
                ),
                "maximum_adjacent_qn_change": max(
                    (abs(value) for value in differences), default=0.0
                ),
                "lambda_qn_nondecreasing": all(
                    value >= -1e-6 for value in penalty_differences
                ),
            }
        )
    switches = []
    distinct = []
    for winners in state_winners.values():
        switches.append(sum(left != right for left, right in zip(winners, winners[1:])))
        distinct.append(len(set(winners)))
    count = len(diagnostics)
    return {
        "candidate_count": count,
        "qn_endpoint_decrease_fraction": fmean(
            item["endpoint_decreased"] for item in diagnostics
        ),
        "qn_nonincreasing_fraction": fmean(
            item["nonincreasing"] for item in diagnostics
        ),
        "qn_direction_reversal_fraction": fmean(
            item["direction_reversals"] > 0 for item in diagnostics
        ),
        "mean_qn_direction_reversals": fmean(
            item["direction_reversals"] for item in diagnostics
        ),
        "median_excess_variation_fraction": median(
            item["excess_variation_fraction"] for item in diagnostics
        ),
        "maximum_adjacent_qn_change": max(
            item["maximum_adjacent_qn_change"] for item in diagnostics
        ),
        "lambda_qn_nondecreasing_fraction": fmean(
            item["lambda_qn_nondecreasing"] for item in diagnostics
        ),
        "state_count": len(state_winners),
        "states_with_policy_switch_fraction": fmean(value > 0 for value in switches),
        "mean_adjacent_policy_switches_per_state": fmean(switches),
        "maximum_distinct_winners_per_state": max(distinct),
    }


def _final_calibration(seed: int, authenticated) -> dict:
    if seed == 0:
        record = authenticated["seed0"]["summary"]["round_records"][-1]
        return record["deployed_after_damping_validation"]
    record = authenticated["summaries"][seed]["round_records"][-1]
    return record["deployed_validation"]


def run(project_root: Path, output_dir: Path, *, device_name: str, max_steps: int):
    torch.set_num_threads(1)
    device = pilot._device(device_name)
    authenticated = _authenticate(project_root, max_steps=max_steps)
    bank, probe_runs = _collect_bank(project_root, authenticated, device=device)
    agents = _load_agents(project_root, authenticated, device=device)
    all_values = []
    all_selections = []
    summaries = {}
    for seed, agent in sorted(agents.items()):
        values, selections, sequences, winners = _decompose(agent, bank)
        all_values.extend(values)
        all_selections.extend(selections)
        convergence = (
            authenticated["seed0"]["summary"]["convergence_assessment"]
            if seed == 0
            else authenticated["summaries"][seed]["stability_gate"]
        )
        summaries[str(seed)] = {
            "training_convergence_passed": bool(convergence["passed"]),
            "terminal_calibration": _final_calibration(seed, authenticated),
            "fixed_bank": summarize_sequences(sequences, winners),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    values_path = output_dir / VALUES_NAME
    selections_path = output_dir / SELECTIONS_NAME
    _atomic_csv(values_path, tuple(all_values[0]), all_values)
    _atomic_csv(selections_path, tuple(all_selections[0]), all_selections)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": "post_training_fixed_training_probe_bank_diagnostic",
        "evaluation_panel_opened": False,
        "checkpoint_selection_performed": False,
        "lambda_grid": list(LAMBDA_GRID),
        "bank": {
            "unique_state_count": len(bank),
            "candidate_count": sum(len(item["records"]) for item in bank),
            "maximum_frontiers_per_probe": MAX_FRONTIERS_PER_PROBE,
            "probe_runs": probe_runs,
            "states": [
                {
                    "state_id": item["state_id"],
                    "frontier_digest": item["digest"],
                    "candidate_count": len(item["records"]),
                    "sources": item["sources"],
                }
                for item in bank
            ],
        },
        "seeds": summaries,
        "provenance": {
            "seed0_terminal_sha256": authenticated["seed0"]["terminal_sha256"],
            "seeds12_report_sha256": authenticated["seeds12_report_sha256"],
            "seed1_terminal_sha256": _sha256(
                authenticated["seeds12_root"] / "seed-1" / "terminal.pth"
            ),
            "seed2_terminal_sha256": _sha256(
                authenticated["seeds12_root"] / "seed-2" / "terminal.pth"
            ),
            "diagnostic_source_sha256": _sha256(Path(__file__).resolve()),
        },
        "artifacts": {
            "values_csv": str(values_path.resolve()),
            "selections_csv": str(selections_path.resolve()),
        },
        "interpretation_rule": {
            "smooth_qn_decrease": (
                "consistent with learned policy adaptation at fixed state/candidate"
            ),
            "qn_direction_reversals_or_large_adjacent_changes": (
                "evidence of lambda-conditioning fit/calibration irregularity"
            ),
            "policy_switch_with_smooth_components": (
                "a merit-ranking crossover, not by itself fitting noise"
            ),
        },
    }
    _atomic_json(output_dir / REPORT_NAME, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-steps", type=int, default=2_000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_steps <= 0:
        raise ValueError("max steps must be positive")
    project_root = args.project_root.resolve()
    output_dir = (
        project_root / OUTPUT_RELATIVE
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    report = run(
        project_root,
        output_dir,
        device_name=args.device,
        max_steps=args.max_steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
