# Conditioned VCG development history

These screens explain how the final method was selected. They are supporting
development evidence, not the paper's E-series benchmark experiments.

| ID | Experiment | Outcome | Frozen protocol location |
|---|---|---|---|
| D1 | Joint-vector architecture screen | rejected: operational endpoint not preserved | [`vcg_preference_conditioned_architecture_screen_85k/`](../../vcg_preference_conditioned_architecture_screen_85k/) |
| D2 | Anchored operational-residual screen | rejected: high-lambda handling failure | [`vcg_v11_anchored_preference_seed0_85k/`](../../vcg_v11_anchored_preference_seed0_85k/) |
| D3 | Conditioned future-handling seed-0 screen | advanced | [`vcg_v11_conditioned_handling_seed0_85k/`](../../vcg_v11_conditioned_handling_seed0_85k/) |
| D4 | Full-update convergence extension | rejected by convergence diagnostic | [`vcg_v11_conditioned_handling_convergence_extension/`](../../vcg_v11_conditioned_handling_convergence_extension/) |
| D5 | Damped seed-0 convergence | passed at round 8 | [`vcg_v11_conditioned_handling_damped_convergence/`](../../vcg_v11_conditioned_handling_damped_convergence/) |
| D6 | Frozen damped seed-0 evaluation | advanced | [`vcg_v11_conditioned_handling_damped_evaluation_85k/`](../../vcg_v11_conditioned_handling_damped_evaluation_85k/) |
| D7 | Matched seed-1/2 replication | seed 2 passed; seed 1 continued | [`vcg_v11_conditioned_handling_two_phase_seeds12/`](../../vcg_v11_conditioned_handling_two_phase_seeds12/) |
| D8 | Seed-1 convergence continuation | passed at round 10 | [`vcg_v11_conditioned_handling_seed1_convergence_continuation/`](../../vcg_v11_conditioned_handling_seed1_convergence_continuation/) |
| D9 | Fixed merit-bank diagnostic | diagnostic complete | [`vcg_v11_conditioned_handling_fixed_merit_bank/`](../../vcg_v11_conditioned_handling_fixed_merit_bank/) |
| [D10](D10_scalability_support_screen/) | Zero-shot scale support and instrumentation | v2 passed 24/24 through 8×8; 10×10 low completed and medium/high latency pilots were right-censored at one hour | local staged screen |
| [D11](D11_shared_search_opportunity_audit/) | Shared-search opportunity audit | bounded intermediate-state overlap and net transition-reuse measurement | local staged screen |

Each D-directory contains a thin launcher with the same arguments and
environment variables as its frozen protocol launcher.
