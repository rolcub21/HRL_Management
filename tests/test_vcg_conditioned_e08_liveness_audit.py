from experiments.conditioned_vcg.E08_liveness_audit_95k import (
    analyze_existing,
    instrumented_pilot,
    summarize_pilot,
)


def test_e8_existing_e13_aggregate_is_complete():
    rows, _sources = analyze_existing._e13_rows()
    summary, by_scenario = analyze_existing._aggregate_e13(rows)

    assert len(rows) == 45
    assert len(by_scenario) == 15
    assert summary["strict_safe_complete_episodes"] == 45
    assert summary["macro_decisions"] == 1747
    assert summary["liveness_forced_decisions"] == 663
    assert summary["proposal_observed_decisions"] == 0


def test_e8_pilot_is_bound_to_guard_heavy_frozen_coordinate():
    parent = instrumented_pilot._frozen_parent()
    row = parent["ledger"]["row"]

    assert row["scenario_id"] == "size_8x8_occ_high"
    assert row["instance_seed"] == 95100
    assert row["macro_decisions"] == 42
    assert sum(item["liveness_forced"] for item in row["decision_costs"]) == 27


def test_e8_forced_run_summary_does_not_call_runs_activations():
    decisions = [
        {"liveness_forced": False, "selected_action_type": "defer"},
        {"liveness_forced": True, "selected_action_type": "reconfigure"},
        {"liveness_forced": True, "selected_action_type": "deliver"},
        {"liveness_forced": False, "selected_action_type": "accept"},
    ]

    runs, preceding = analyze_existing._forced_runs(decisions)

    assert len(runs) == 1
    assert runs[0]["length"] == 2
    assert preceding == {"defer": 1}


def test_e8_trace_exposes_counter_cursor_and_admission_mask():
    report = summarize_pilot._load_report(
        instrumented_pilot.DEFAULT_OUTPUT / summarize_pilot.INPUT_NAME
    )
    decisions = report["decisions"]
    runs = summarize_pilot._forced_runs(decisions)

    assert runs == [
        {
            "start_decision": 4,
            "end_decision_inclusive": 30,
            "length": 27,
            "activation_reason": "nonprogress_limit",
            "witness_length": 27,
        }
    ]
    forced = decisions[4:31]
    assert [item["active_witness_cursor_after_select"] for item in forced] == list(
        range(27)
    )
    assert forced[0]["nonprogress_count_before_select"] == 2
    assert forced[0]["nonprogress_trigger"] is True
    assert sum(
        item["learned_proposal_action_type"] == "accept" for item in forced
    ) == 16
    assert all(item["executed_mode"] == "recover" for item in forced)
    assert decisions[31]["guard_active_before_select"] is False
    assert decisions[31]["active_witness_length_after_select"] == 0
    assert decisions[31]["executed_action_type"] == "accept"
