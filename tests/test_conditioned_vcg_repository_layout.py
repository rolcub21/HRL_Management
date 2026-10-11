from pathlib import Path

import methods.conditioned_vcg as public
import vcg_v11_conditioned_handling as frozen


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "experiments" / "conditioned_vcg"
DEVELOPMENT = PROGRAM / "development"
PAPER_EXPERIMENTS = {
    "E01_benchmark_90k": (
        "experiments/vcg_conditioned_final_comparison_90k/run.sh"
    ),
    "E03_certification_ablation_90k": None,
    "E04_safe_frontier_ranking_92k": None,
    "E05_handling_model_ablation_92k": None,
    "E08_liveness_audit_95k": None,
    "E11_distribution_shift_93k": None,
    "E12_representation_ablation_94k": None,
    "E13_operational_scalability_95k": None,
    "E14_certification_scalability_95k": None,
    "E16_prediction_ranking_audit": None,
    "E19_service_tails_cost_sensitivity": None,
}
DEVELOPMENT_EXPERIMENTS = {
    "D01_architecture_screen": (
        "experiments/vcg_preference_conditioned_architecture_screen_85k/run.sh"
    ),
    "D02_operational_anchor_screen": (
        "experiments/vcg_v11_anchored_preference_seed0_85k/run.sh"
    ),
    "D03_conditioned_seed0_screen": (
        "experiments/vcg_v11_conditioned_handling_seed0_85k/run.sh"
    ),
    "D04_full_update_convergence": (
        "experiments/vcg_v11_conditioned_handling_convergence_extension/run.sh"
    ),
    "D05_damped_convergence": (
        "experiments/vcg_v11_conditioned_handling_damped_convergence/run.sh"
    ),
    "D06_damped_seed0_evaluation": (
        "experiments/vcg_v11_conditioned_handling_damped_evaluation_85k/run.sh"
    ),
    "D07_seed_replication": (
        "experiments/vcg_v11_conditioned_handling_two_phase_seeds12/run.sh"
    ),
    "D08_seed1_continuation": (
        "experiments/vcg_v11_conditioned_handling_seed1_convergence_continuation/run.sh"
    ),
    "D09_fixed_merit_diagnostic": (
        "experiments/vcg_v11_conditioned_handling_fixed_merit_bank/run.sh"
    ),
    "D10_scalability_support_screen": None,
    "D11_shared_search_opportunity_audit": None,
    "D12_relocation_family_certification": None,
}


def test_public_method_package_reexports_frozen_evidence_implementation():
    assert public.ConditionedHandlingAgent is frozen.ConditionedHandlingAgent
    assert public.ConditionedHandlingConfig is frozen.ConditionedHandlingConfig
    assert public.CONTROLLER_ARCHITECTURE == frozen.CONTROLLER_ARCHITECTURE
    assert public.TARGET_CONTRACT == frozen.TARGET_CONTRACT


def test_paper_experiment_program_is_separate_and_ordered():
    observed = sorted(
        path.name for path in PROGRAM.iterdir() if path.is_dir()
    )
    assert observed == [*PAPER_EXPERIMENTS, "development"]


def test_development_history_is_complete_and_ordered():
    observed = sorted(path.name for path in DEVELOPMENT.iterdir() if path.is_dir())
    assert observed == list(DEVELOPMENT_EXPERIMENTS)


def test_numbered_launchers_delegate_to_regular_frozen_launchers():
    programs = (
        (PROGRAM, PAPER_EXPERIMENTS),
        (DEVELOPMENT, DEVELOPMENT_EXPERIMENTS),
    )
    for base, experiments in programs:
        for directory, target in experiments.items():
            launcher = base / directory / "run.sh"
            readme = base / directory / "README.md"
            assert launcher.is_file() and not launcher.is_symlink()
            assert readme.is_file()
            if target is None:
                local_programs = tuple((base / directory).glob("*.py"))
                launcher_text = launcher.read_text(encoding="utf-8")
                assert local_programs
                assert any(
                    program.name in launcher_text
                    or program.stem in launcher_text
                    for program in local_programs
                )
                continue
            frozen_launcher = ROOT / target
            assert frozen_launcher.is_file() and not frozen_launcher.is_symlink()
            assert target in launcher.read_text(encoding="utf-8")
