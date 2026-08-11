import contextlib
import io
import math
from pathlib import Path
from statistics import fmean, stdev
import tempfile
import unittest

import torch

import train_vcg_constrained_v2_3 as v23


def _validation_row(instance_index, rng_index, *, physical=0, mae=3.0):
    seed = v23.DEFAULT_VALIDATION_SEEDS[instance_index]
    return {
        "strict_method_success": True,
        "physical_rehandles": int(physical),
        "required_deliveries": 5,
        "delivery_count": 5,
        "dense_return": 100.0,
        "mean_absolute_error": float(mae),
        "method_failure_reason": None,
        "illegal_drops": 0,
        "fallbacks": 0,
        "witness_mismatches": 0,
        "all_selected_candidates_exact_safe": True,
        "instance_seed": seed,
        "instance_index": instance_index,
        "episode_instance_id": f"instance-{seed}",
        "schedule_id": f"schedule-{seed}",
        "episode_instance_sha256": f"{seed:064x}",
        "policy_rng_index": rng_index,
        "policy_rng_seed": v23.validation_policy_rng_seed(
            instance_index, rng_index
        ),
        "evaluation_learning": False,
        "training_agent_unchanged": True,
        "validation_batch_state_unchanged": True,
        "fresh_evaluation_clone": True,
        "stochastic_selection_only": True,
        "macro_decisions": 1,
        "selected_action_counts": {"deliver": 1},
        "hold_outcome_counts": {},
        "policy_diagnostic_decisions": 1,
        "mean_outer_policy_entropy": 0.1,
        "mean_outer_map_probability": 0.9,
        "mean_selected_within_policy_entropy": 0.05,
        "mean_selected_within_map_probability": 0.95,
    }


def _validation_grid():
    rows = []
    for instance_index in range(len(v23.DEFAULT_VALIDATION_SEEDS)):
        # Cluster rates repeat 0, 5, 10, and 15 per 100 required blocks.
        physical_count = instance_index % 4
        for rng_index in range(v23.VALIDATION_POLICY_RNG_COUNT):
            rows.append(
                _validation_row(
                    instance_index,
                    rng_index,
                    physical=int(rng_index < physical_count),
                )
            )
    return rows


class V23FrozenProtocolTests(unittest.TestCase):
    def test_defaults_schedule_and_seed_namespaces_are_exact(self):
        with tempfile.TemporaryDirectory() as parent:
            args = v23.build_parser().parse_args(
                ["--output-dir", str(Path(parent) / "contract")]
            )
            contract = v23.build_training_contract(args)

        self.assertEqual(args.model_seed, 10)
        self.assertEqual(args.episodes, 200)
        self.assertEqual(args.validation_every, 20)
        self.assertEqual(args.max_steps, 2_000)
        self.assertEqual(args.gamma_operational, 1.0)
        self.assertEqual(tuple(args.validation_seeds), tuple(range(85_000, 85_012)))
        self.assertEqual(contract["validation_rows_per_checkpoint"], 48)
        self.assertEqual(
            contract["validation_design"]["statistical_unit"],
            "EpisodeInstance_cluster",
        )
        self.assertEqual(
            tuple(contract["validation_design"]["candidate_looks"]),
            v23.CANDIDATE_EPISODES,
        )
        self.assertFalse(contract["validation_design"]["final_panel_opened"])
        self.assertEqual(contract["gamma_operational"], 1.0)
        self.assertEqual(contract["gamma_rehandle"], 1.0)
        self.assertEqual(contract["agent_config"]["gamma_operational"], 1.0)
        self.assertTrue(contract["development_panel_reused_after_v2_2"])
        self.assertTrue(contract["training_instances_reused_after_v2_2"])
        self.assertFalse(contract["fresh_prospective_experiment"])
        self.assertEqual(
            contract["isolated_algorithmic_change"],
            {
                "parameter": "gamma_operational",
                "parent_value": 0.99,
                "ablation_value": 1.0,
                "all_other_algorithmic_and_protocol_settings_frozen_to_parent": True,
            },
        )
        self.assertEqual(
            contract["parent_v2_2_binding"]["training_contract_canonical_sha256"],
            v23.PARENT_V2_2_CONTRACT_SHA256,
        )
        self.assertEqual(
            contract["parent_v2_2_binding"]["validation_manifest_canonical_sha256"],
            v23.PARENT_V2_2_VALIDATION_MANIFEST_SHA256,
        )

        expected_temperatures = {
            4: (0.06309573444801933, 0.5492802716530588),
            5: (0.03981071705534972, 0.30170881682725814),
            6: (0.0251188643150958, 0.16572270086699936),
            7: (0.01584893192461113, 0.09102821015130401),
        }
        for block, (within, group) in expected_temperatures.items():
            observed_within, observed_group = v23.temperatures_for_block(block)
            self.assertEqual(observed_within, (within,) * 4)
            self.assertEqual(observed_group, group)
        self.assertEqual(v23.temperatures_for_block(20), ((0.01,) * 4, 0.05))

        self.assertEqual(v23.training_policy_rng_seed(1), 610_000_000)
        self.assertEqual(v23.training_policy_rng_seed(200), 610_000_199)
        self.assertEqual(v23.validation_policy_rng_seed(0, 0), 620_000_000)
        self.assertEqual(v23.validation_policy_rng_seed(11, 3), 620_000_047)
        self.assertLess(620_000_047, v23.FINAL_POLICY_RNG_BASE)

    def test_contract_rejects_development_and_final_panel_leakage(self):
        with tempfile.TemporaryDirectory() as parent:
            parser = v23.build_parser()
            for seeds in ((83_000,) * 12, tuple(range(86_000, 86_012))):
                argv = ["--output-dir", str(Path(parent) / str(seeds[0]))]
                argv += ["--validation-seeds", *(str(seed) for seed in seeds)]
                with self.subTest(first_seed=seeds[0]):
                    with self.assertRaisesRegex(
                        v23.ConstrainedV23TrainerError, "frozen 85000"
                    ):
                        v23.build_training_contract(parser.parse_args(argv))

    def test_parent_gamma_cannot_be_selected_through_cli(self):
        with tempfile.TemporaryDirectory() as parent:
            args = v23.build_parser().parse_args([
                "--output-dir",
                str(Path(parent) / "wrong-gamma"),
                "--gamma-operational",
                "0.99",
            ])
            with self.assertRaisesRegex(
                v23.ConstrainedV23TrainerError,
                "gamma_operational=1.0",
            ):
                v23.build_training_contract(args)

    def test_ucb_uses_twelve_episode_instance_cluster_means(self):
        rows = _validation_grid()
        summary = v23.summarize_validation(
            rows,
            v23.V23DualConfig(),
            checkpoint_episode=80,
            validation_lambda=0.0,
            schedule_state=v23.schedule_for_episode(80, validation=True).to_dict(),
        )
        cluster_rates = [0.0, 5.0, 10.0, 15.0] * 3
        expected_mean = fmean(cluster_rates)
        expected_se = stdev(cluster_rates) / math.sqrt(12)
        expected_ucb = expected_mean + v23.VALIDATION_COST_T_95 * expected_se
        self.assertEqual(
            tuple(summary["episode_instance_cluster_rehandle_rates"]),
            tuple(cluster_rates),
        )
        self.assertAlmostEqual(
            summary["expected_physical_rehandles_per_100_required_deliveries"],
            expected_mean,
        )
        self.assertAlmostEqual(
            summary["physical_rehandle_rate_cluster_standard_error"], expected_se
        )
        self.assertAlmostEqual(
            summary["physical_rehandle_rate_one_sided_95_ucb"], expected_ucb
        )
        self.assertTrue(summary["development_candidate_eligible"])

        # Historical saturation is diagnostic only. A recovered, currently
        # feasible stochastic policy remains eligible under the frozen rule.
        recovered = v23.summarize_validation(
            rows,
            v23.V23DualConfig(),
            checkpoint_episode=80,
            validation_lambda=1.0,
            schedule_state=v23.schedule_for_episode(80, validation=True).to_dict(),
            positive_residual_saturation_count=3,
        )
        self.assertEqual(
            recovered["positive_residual_dual_saturation_count_before_validation"], 3
        )
        self.assertFalse(recovered["positive_residual_dual_saturation"])
        self.assertTrue(recovered["development_candidate_eligible"])

    def test_candidate_looks_and_all_row_integrity_fail_closed(self):
        rows = _validation_grid()
        early = v23.summarize_validation(
            rows,
            v23.V23DualConfig(),
            checkpoint_episode=60,
            validation_lambda=0.0,
            schedule_state=v23.schedule_for_episode(60, validation=True).to_dict(),
        )
        self.assertFalse(early["candidate_look_gate"])
        self.assertFalse(early["development_candidate_eligible"])

        corrupted = [dict(row) for row in rows]
        corrupted[17]["training_agent_unchanged"] = False
        corrupted[18]["validation_batch_state_unchanged"] = False
        failed = v23.summarize_validation(
            corrupted,
            v23.V23DualConfig(),
            checkpoint_episode=80,
            validation_lambda=0.0,
            schedule_state=v23.schedule_for_episode(80, validation=True).to_dict(),
        )
        self.assertFalse(failed["strict_integrity_gate"])
        self.assertFalse(failed["development_candidate_eligible"])
        with self.assertRaisesRegex(v23.ConstrainedV23TrainerError, "exactly 48"):
            v23.summarize_validation(
                rows[:-1],
                v23.V23DualConfig(),
                checkpoint_episode=80,
                validation_lambda=0.0,
                schedule_state=v23.schedule_for_episode(80, validation=True).to_dict(),
            )


class _FakeRuntime:
    def __init__(self, args):
        from viability_graph_constrained_v2_3 import (
            ConstrainedV23Config,
            ConstrainedV23HierarchyAgent,
        )

        self.args = args
        self.dual_lambda = 0.0
        self.schedule_state = {}
        self.events = []
        self.core_contract = v23._trusted_runtime_core_contract(
            {"agent_config": v23.frozen_agent_config(args)}
        )
        self._checkpoint_agent = ConstrainedV23HierarchyAgent(
            config=ConstrainedV23Config.from_dict(self.core_contract["agent_config"]),
            seed=args.model_seed,
            device="cpu",
            epsilon=0.0,
            dual_lambda=0.0,
        )

    def set_dual_lambda(self, value):
        self.dual_lambda = float(value)
        self._checkpoint_agent.set_dual_lambda(value)

    def set_schedule_state(self, state):
        self.schedule_state = dict(state)
        self._checkpoint_agent.set_schedule_state(state)

    def begin_validation_batch(self):
        self.events.append({"validation_batch": "begin"})
        return {
            "validation_batch_started": True,
            "full_training_state_digest": "fake-digest",
        }

    def end_validation_batch(self):
        self.events.append({"validation_batch": "end"})
        return {
            "validation_batch_ended": True,
            "training_agent_unchanged": True,
            "full_training_state_digest_before": "fake-digest",
            "full_training_state_digest_after": "fake-digest",
        }

    def run_episode(
        self,
        *,
        instance_seed,
        training,
        max_steps,
        policy_rng_index=None,
        policy_rng_seed=None,
    ):
        del max_steps
        self.events.append(
            {
                "training": bool(training),
                "instance_seed": int(instance_seed),
                "block": self.schedule_state["block_number"],
                "lambda": self.dual_lambda,
                "policy_rng_index": policy_rng_index,
                "policy_rng_seed": policy_rng_seed,
            }
        )
        parent_identity = None
        if not training and int(instance_seed) in v23.DEFAULT_VALIDATION_SEEDS:
            parent_identity = v23.PARENT_V2_2_VALIDATION_INSTANCE_IDENTITIES[
                v23.DEFAULT_VALIDATION_SEEDS.index(int(instance_seed))
            ]
        result = {
            "strict_method_success": True,
            "physical_rehandles": 2 if training else 0,
            "required_deliveries": 8,
            "delivery_count": 8,
            "dense_return": 100.0,
            "mean_absolute_error": 3.0,
            "method_failure_reason": None,
            "illegal_drops": 0,
            "fallbacks": 0,
            "witness_mismatches": 0,
            "all_selected_candidates_exact_safe": True,
            "baseline_teacher": False,
            "baseline_policy_query": False,
            "method_version": v23.METHOD_VERSION,
            "policy_schedule_protocol": v23.POLICY_SCHEDULE_PROTOCOL,
            "policy_mode": v23.POLICY_REALIZATION,
            "execution_context": self.schedule_state["execution_context"],
            "within_group_temperatures": tuple(
                self.schedule_state["within_group_temperatures"]
            ),
            "group_temperature": self.schedule_state["group_temperature"],
            "dual_lambda": self.dual_lambda,
            "map_selection_used": False,
            "macro_decisions": 1,
            "selected_action_counts": {"deliver": 1},
            "hold_outcome_counts": {},
            "policy_diagnostic_decisions": 1,
            "mean_outer_policy_entropy": 0.1,
            "mean_outer_map_probability": 0.9,
            "mean_selected_within_policy_entropy": 0.05,
            "mean_selected_within_map_probability": 0.95,
            "policy_rng_index": policy_rng_index,
            "policy_rng_seed": policy_rng_seed,
            "policy_rng_scope": "action_sampling_only",
            "instance_seed": int(instance_seed),
            "episode_instance_id": (
                f"instance-{instance_seed}"
                if parent_identity is None
                else parent_identity["episode_instance_id"]
            ),
            "schedule_id": (
                f"schedule-{instance_seed}"
                if parent_identity is None
                else parent_identity["schedule_id"]
            ),
            "episode_instance_sha256": (
                f"{int(instance_seed):064x}"
                if parent_identity is None
                else parent_identity["episode_instance_sha256"]
            ),
            "replay_rng_seed": v23.REPLAY_RNG_SEED,
            "behavior_policy_rng_reset": True,
        }
        if not training:
            result.update(
                evaluation_learning=False,
                training_agent_unchanged=True,
                fresh_evaluation_clone=True,
                stochastic_selection_only=True,
                option_evaluation_mode=True,
            )
        return result

    def checkpoint_state(self, *, include_replay):
        if not include_replay:
            return self._checkpoint_agent.checkpoint(include_replay=False)
        return {
            "checkpoint_family": v23.CHECKPOINT_FAMILY,
            "config": v23.frozen_agent_config(self.args),
            "agent_state": {
                "Q_local": {"fake_weight": torch.tensor([1.0])},
                "Q_target": {"fake_weight": torch.tensor([1.0])},
                "optimizer": {"fake": True},
                "replay": {"fake": True} if include_replay else None,
                "dual_lambda": self.dual_lambda,
                "schedule_state": dict(self.schedule_state),
                "replay_rng_seed": v23.REPLAY_RNG_SEED,
                "rng_state": ("fake",),
            },
        }


class V23CadenceAndCheckpointTests(unittest.TestCase):
    def test_transient_per_row_training_state_mutation_fails_closed(self):
        class TransientMutationRuntime(_FakeRuntime):
            emitted = False

            def run_episode(self, **kwargs):
                row = super().run_episode(**kwargs)
                if not kwargs["training"] and not self.emitted:
                    self.emitted = True
                    row["training_agent_unchanged"] = False
                return row

        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "transient-mutation"
            args = v23.build_parser().parse_args(["--output-dir", str(output)])
            with self.assertRaisesRegex(RuntimeError, "mutated and later restored"):
                with contextlib.redirect_stdout(io.StringIO()):
                    v23.run_development_calibration(
                        args, runtime=TransientMutationRuntime(args)
                    )

    def test_dual_updates_on_nonvalidation_blocks_and_only_best_is_candidate(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "fake-v23"
            args = v23.build_parser().parse_args(["--output-dir", str(output)])
            runtime = _FakeRuntime(args)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = v23.run_development_calibration(args, runtime=runtime)

            self.assertEqual(summary["completed_training_episodes"], 200)
            self.assertEqual(summary["dual_state"]["update_count"], 17)
            self.assertEqual(len(summary["dual_update_history"]), 18)
            self.assertFalse(summary["dual_update_history"][-1]["applied"])
            self.assertEqual(
                [item["checkpoint_episode"] for item in summary["validation_history"]],
                list(range(20, 201, 20)),
            )

            validation_events = [
                event
                for event in runtime.events
                if "training" in event and not event["training"]
            ]
            first_ep40 = validation_events[48]
            self.assertEqual(first_ep40["block"], 4)
            # Block 3 updates lambda at ep30 even though there is no validation.
            self.assertGreater(first_ep40["lambda"], 0.0)
            self.assertTrue(
                all(
                    event["instance_seed"] not in v23.FINAL_PANEL_SEEDS
                    for event in runtime.events
                    if "instance_seed" in event
                )
            )

            best = torch.load(
                output / "best-development-candidate.pth",
                map_location="cpu",
                weights_only=False,
            )
            latest = torch.load(
                output / "latest.pth", map_location="cpu", weights_only=False
            )
            self.assertEqual(best["checkpoint_role"], "best_development_candidate")
            self.assertEqual(best["completed_episodes"], 80)
            self.assertTrue(best["development_candidate_eligible"])
            self.assertTrue(best["validation_authenticates_checkpoint_lambda"])
            self.assertEqual(best["pending_dual_batch"]["episode_count"], 10)

            self.assertEqual(latest["checkpoint_role"], "latest_development_state")
            self.assertFalse(latest["development_candidate_eligible"])
            self.assertFalse(latest["deployment_checkpoint_eligible"])
            self.assertFalse(latest["terminal_dual_proposal"]["applied"])

            diagnostic_records = summary["candidate_look_diagnostic_checkpoints"]
            self.assertEqual(
                [record["checkpoint_episode"] for record in diagnostic_records],
                list(v23.CANDIDATE_EPISODES),
            )
            self.assertTrue(all(len(record["sha256"]) == 64 for record in diagnostic_records))
            diagnostic = torch.load(
                output / diagnostic_records[0]["relative_path"],
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(diagnostic["checkpoint_role"], "candidate_look_diagnostic")
            self.assertTrue(diagnostic["model_only"])
            self.assertTrue(diagnostic["diagnostic_only"])
            self.assertFalse(diagnostic["development_candidate_eligible"])
            self.assertFalse(diagnostic["deployment_checkpoint_eligible"])
            model_checkpoint = diagnostic["agent_state"]
            self.assertTrue(model_checkpoint["model_only"])
            self.assertEqual(
                set(model_checkpoint["agent_state"]),
                set(v23._MODEL_ONLY_AGENT_STATE_KEYS),
            )
            self.assertNotIn("optimizer", model_checkpoint["agent_state"])
            self.assertNotIn("replay", model_checkpoint["agent_state"])

            import json

            manifest = json.loads(
                Path(summary["candidate_look_checkpoint_manifest"]).read_text()
            )
            claimed_hash = manifest.pop("manifest_sha256")
            self.assertEqual(v23.contract_hash(manifest), claimed_hash)
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["artifact_count"], len(v23.CANDIDATE_EPISODES))
            authenticated = v23.load_candidate_look_diagnostic(
                output / diagnostic_records[0]["relative_path"],
                manifest_path=summary["candidate_look_checkpoint_manifest"],
                contract=output / "training-contract.json",
                device="cpu",
            )
            self.assertEqual(
                authenticated["checkpoint"]["completed_episodes"],
                v23.CANDIDATE_EPISODES[0],
            )

            best_path = output / "best-development-candidate.pth"
            expected_best_sha = v23._file_sha256(best_path)
            authenticated_best = v23.load_best_development_candidate(
                best_path,
                expected_best_sha256=expected_best_sha,
                manifest_path=summary["candidate_look_checkpoint_manifest"],
                expected_manifest_sha256=claimed_hash,
                validation_instance_manifest_path=summary[
                    "validation_instance_manifest"
                ],
                contract=summary["training_contract"],
                device="cpu",
            )
            self.assertEqual(authenticated_best["selected_episode"], 80)
            self.assertEqual(
                len(authenticated_best["candidate_records"]),
                len(v23.CANDIDATE_EPISODES),
            )
            self.assertEqual(
                [
                    record["validation_summary"]["checkpoint_episode"]
                    for record in authenticated_best["candidate_records"]
                ],
                list(v23.CANDIDATE_EPISODES),
            )
            self.assertIsNone(authenticated_best["agent"].behavior_rng_seed)

            with self.assertRaisesRegex(
                v23.ConstrainedV23TrainerError, "raw file SHA"
            ):
                v23.load_best_development_candidate(
                    best_path,
                    expected_best_sha256="0" * 64,
                    manifest_path=summary["candidate_look_checkpoint_manifest"],
                    expected_manifest_sha256=claimed_hash,
                    validation_instance_manifest_path=summary[
                        "validation_instance_manifest"
                    ],
                    contract=summary["training_contract"],
                    device="cpu",
                )
            with self.assertRaisesRegex(
                v23.ConstrainedV23TrainerError, "externally audited SHA"
            ):
                v23.load_best_development_candidate(
                    best_path,
                    expected_best_sha256=expected_best_sha,
                    manifest_path=summary["candidate_look_checkpoint_manifest"],
                    expected_manifest_sha256="0" * 64,
                    validation_instance_manifest_path=summary[
                        "validation_instance_manifest"
                    ],
                    contract=summary["training_contract"],
                    device="cpu",
                )

            original_best = torch.load(
                best_path, map_location="cpu", weights_only=True
            )
            wrong_role = dict(original_best)
            wrong_role["checkpoint_role"] = "latest_development_state"
            torch.save(wrong_role, best_path)
            with self.assertRaisesRegex(
                v23.ConstrainedV23TrainerError, "checkpoint envelope mismatch"
            ):
                v23.load_best_development_candidate(
                    best_path,
                    expected_best_sha256=v23._file_sha256(best_path),
                    manifest_path=summary["candidate_look_checkpoint_manifest"],
                    expected_manifest_sha256=claimed_hash,
                    validation_instance_manifest_path=summary[
                        "validation_instance_manifest"
                    ],
                    contract=summary["training_contract"],
                    device="cpu",
                )

            nested_extra = dict(original_best)
            nested_extra["agent_state"] = dict(original_best["agent_state"])
            nested_extra["agent_state"]["replay"] = {"injected": True}
            torch.save(nested_extra, best_path)
            with self.assertRaisesRegex(
                v23.ConstrainedV23TrainerError,
                "nested best-development checkpoint envelope schema",
            ):
                v23.load_best_development_candidate(
                    best_path,
                    expected_best_sha256=v23._file_sha256(best_path),
                    manifest_path=summary["candidate_look_checkpoint_manifest"],
                    expected_manifest_sha256=claimed_hash,
                    validation_instance_manifest_path=summary[
                        "validation_instance_manifest"
                    ],
                    contract=summary["training_contract"],
                    device="cpu",
                )

            lambda_drift = dict(original_best)
            lambda_drift["runtime_lambda"] += 0.125
            torch.save(lambda_drift, best_path)
            with self.assertRaisesRegex(
                v23.ConstrainedV23TrainerError, "lambda authentication"
            ):
                v23.load_best_development_candidate(
                    best_path,
                    expected_best_sha256=v23._file_sha256(best_path),
                    manifest_path=summary["candidate_look_checkpoint_manifest"],
                    expected_manifest_sha256=claimed_hash,
                    validation_instance_manifest_path=summary[
                        "validation_instance_manifest"
                    ],
                    contract=summary["training_contract"],
                    device="cpu",
                )

            schedule_drift = dict(original_best)
            schedule_drift["schedule_state"] = dict(original_best["schedule_state"])
            schedule_drift["schedule_state"]["block_number"] += 1
            torch.save(schedule_drift, best_path)
            with self.assertRaisesRegex(
                v23.ConstrainedV23TrainerError, "schedule authentication"
            ):
                v23.load_best_development_candidate(
                    best_path,
                    expected_best_sha256=v23._file_sha256(best_path),
                    manifest_path=summary["candidate_look_checkpoint_manifest"],
                    expected_manifest_sha256=claimed_hash,
                    validation_instance_manifest_path=summary[
                        "validation_instance_manifest"
                    ],
                    contract=summary["training_contract"],
                    device="cpu",
                )

            model_drift = dict(original_best)
            model_drift["agent_state"] = dict(original_best["agent_state"])
            model_drift["agent_state"]["agent_state"] = dict(
                original_best["agent_state"]["agent_state"]
            )
            model_drift["agent_state"]["agent_state"]["Q_local"] = dict(
                original_best["agent_state"]["agent_state"]["Q_local"]
            )
            parameter_name = next(
                iter(model_drift["agent_state"]["agent_state"]["Q_local"])
            )
            parameter = model_drift["agent_state"]["agent_state"]["Q_local"][
                parameter_name
            ].clone()
            parameter.view(-1)[0] += 1.0
            model_drift["agent_state"]["agent_state"]["Q_local"][
                parameter_name
            ] = parameter
            torch.save(model_drift, best_path)
            with self.assertRaisesRegex(
                v23.ConstrainedV23TrainerError, "model differs"
            ):
                v23.load_best_development_candidate(
                    best_path,
                    expected_best_sha256=v23._file_sha256(best_path),
                    manifest_path=summary["candidate_look_checkpoint_manifest"],
                    expected_manifest_sha256=claimed_hash,
                    validation_instance_manifest_path=summary[
                        "validation_instance_manifest"
                    ],
                    contract=summary["training_contract"],
                    device="cpu",
                )

            # Rebind every mutable integrity field around episode 100 to make
            # it the apparent higher-return eligible look. The external best
            # SHA still authenticates the genuine episode-80 checkpoint, so
            # recomputing all seven ledger summaries must reject the stale best.
            torch.save(original_best, best_path)
            rebound_manifest = json.loads(
                Path(summary["candidate_look_checkpoint_manifest"]).read_text()
            )
            episode = 100
            ledger_path = output / f"validation-ledger/episode-{episode:04d}.json"
            ledger = json.loads(ledger_path.read_text())
            for row in ledger["rows"]:
                row["dense_return"] = 200.0
            ledger.pop("ledger_sha256")
            ledger["ledger_sha256"] = v23.contract_hash(ledger)
            ledger_path.write_text(json.dumps(ledger))

            artifact = next(
                item
                for item in rebound_manifest["artifacts"]
                if item["checkpoint_episode"] == episode
            )
            diagnostic_path = output / artifact["relative_path"]
            diagnostic_payload = torch.load(
                diagnostic_path, map_location="cpu", weights_only=True
            )
            old_validation = diagnostic_payload["validation_summary"]
            recomputed = v23.summarize_validation(
                ledger["rows"],
                v23.V23DualConfig(),
                checkpoint_episode=episode,
                validation_lambda=old_validation["validation_lambda"],
                schedule_state=old_validation["schedule_state"],
                positive_residual_saturation_count=old_validation[
                    "positive_residual_dual_saturation_count_before_validation"
                ],
            )
            recomputed["validated_before_dual_update"] = True
            recomputed["validation_batch_audit"] = ledger[
                "validation_batch_audit"
            ]
            recomputed["validation_ledger"] = {
                "relative_path": f"validation-ledger/episode-{episode:04d}.json",
                "ledger_sha256": ledger["ledger_sha256"],
                "row_count": ledger["row_count"],
            }
            diagnostic_payload["validation_summary"] = recomputed
            torch.save(diagnostic_payload, diagnostic_path)
            artifact["sha256"] = v23._file_sha256(diagnostic_path)
            rebound_manifest.pop("manifest_sha256")
            rebound_manifest["manifest_sha256"] = v23.contract_hash(
                rebound_manifest
            )
            Path(summary["candidate_look_checkpoint_manifest"]).write_text(
                json.dumps(rebound_manifest)
            )
            with self.assertRaisesRegex(
                v23.ConstrainedV23TrainerError, "recomputed frozen selection"
            ):
                v23.load_best_development_candidate(
                    best_path,
                    expected_best_sha256=v23._file_sha256(best_path),
                    manifest_path=summary["candidate_look_checkpoint_manifest"],
                    expected_manifest_sha256=rebound_manifest["manifest_sha256"],
                    validation_instance_manifest_path=summary[
                        "validation_instance_manifest"
                    ],
                    contract=summary["training_contract"],
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
