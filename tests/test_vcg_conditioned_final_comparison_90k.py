import run_vcg_conditioned_final_comparison_90k as final90
import train_vcg_preference_conditioned as preference_training


def _safe_row(seed, value=1.0):
    return {
        "instance_seed": seed,
        "strict_safe_complete": True,
        **{metric: value for metric in final90.METRICS},
    }


def test_prospective_grid_is_complete_and_disjoint_from_development():
    assert final90.PROTOCOL.endswith("_cpu_v3")
    assert final90.OPEN_CONFIRMATION.endswith("_CPU_V3")
    assert final90.DEFAULT_OUTPUT.name.endswith("-cpu-v3")
    assert final90.V23_POLICY_RNG_BASE == 622_000_000
    assert tuple(
        range(final90.V23_POLICY_RNG_BASE, final90.V23_POLICY_RNG_BASE + 120)
    ) == final90.v23_adapter.FINAL_POLICY_RNG_SEEDS
    assert final90.INSTANCE_SEEDS == tuple(range(90_000, 90_030))
    assert set(final90.INSTANCE_SEEDS).isdisjoint(
        preference_training.PROTECTED_DEVELOPMENT_AND_TEST_SEEDS
    )
    assert final90.LAMBDA_GRID == (
        0.0,
        0.025,
        0.0375,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.175,
        0.2,
    )
    assert final90.CONDITIONED_ROWS == 900
    assert final90.EXPECTED_ROWS == 1860


def test_aggregate_uses_episode_instance_as_unit():
    rows = [
        _safe_row(seed, value=float(replication + 1))
        for seed in final90.INSTANCE_SEEDS
        for replication in range(3)
    ]
    result = final90._aggregate(rows, expected_per_instance=3)
    assert result["whole_method_eligible"] is True
    assert result["metrics"]["dense_return"] == 2.0
    assert len(result["instance_points"]) == 30


def test_one_failed_cell_suppresses_whole_arm_metrics():
    rows = [
        _safe_row(seed)
        for seed in final90.INSTANCE_SEEDS
        for _replication in range(3)
    ]
    rows[10]["strict_safe_complete"] = False
    rows[10]["method_failure_reason"] = "blocked"
    rows[10]["safety_issues"] = ["blocked"]
    result = final90._aggregate(rows, expected_per_instance=3)
    assert result["whole_method_eligible"] is False
    assert result["numeric_metrics_suppressed"] is True
    assert result["metrics"] is None


def test_lambda_tokens_are_unique():
    tokens = [final90._lambda_token(value) for value in final90.LAMBDA_GRID]
    assert len(tokens) == len(set(tokens))


def test_runtime_contract_does_not_probe_host_runtime(monkeypatch):
    def forbidden_probe():
        raise AssertionError("CPU contract must not probe CUDA or session state")

    monkeypatch.setattr(final90.final86, "_runtime_versions", forbidden_probe)
    assert final90._software_runtime_contract() == {
        "python": final90.platform.python_version(),
        "numpy": str(final90.np.__version__),
        "torch": str(final90.torch.__version__),
        "torch_cuda_build": (
            None
            if final90.torch.version.cuda is None
            else str(final90.torch.version.cuda)
        ),
    }
    provenance = final90._cpu_runtime_provenance()
    assert provenance["execution_device"] == "cpu"
    assert provenance["cuda_runtime_was_probed"] is False
