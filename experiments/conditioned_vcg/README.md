# Conditioned VCG experiments

Paper-facing experiments use zero-padded `E` identifiers. Development screens
use a separate `D` sequence so rejected architectures are not confused with
the experiments that support the final paper.

| ID | Experiment | Role | Status |
|---|---|---|---|
| [E1](E01_benchmark_90k/) | Prospective common-panel benchmark | primary timing--handling result | complete: 1,860/1,860 rows |
| [E2](E01_benchmark_90k/) | Frozen-preference operating points | reports the ten-point timing--handling family from the closed E1 panel | complete: ten predeclared lambda values |
| [E3](E03_certification_ablation_90k/) | Candidate-source certification ablation | tests whether the verifier prevents self-blocking | complete: 90/90 certified, 86/90 physical-only |
| [E4](E04_safe_frontier_ranking_92k/) | Certified-frontier ranking ablation | isolates management quality added by the ranking signal | complete: 540/540 valid rows |
| [E5](E05_handling_model_ablation_92k/) | Handling validation and deployment ablations | validates `Q_N`, its preference input, and the learned future-consequence term | complete: E5(a)--E5(c) |
| [E8](E08_liveness_audit_95k/) | Recovery-witness liveness attribution | separates guard activation, actual override, and learned proposals | complete: log audit and 42-decision instrumented pilot |
| [E11](E11_distribution_shift_93k/) | Frozen-policy distribution-shift generalization | tests completion and timing--handling quality under workload and mirrored-access shifts | complete: 2,730/2,730 rows |
| [E12](E12_representation_ablation_94k/) | Representation attribution under shift | independently retrains relational, nonrelational, and successor-free controllers | complete: 7,560/7,560 rows; 7,557 strict |
| [E13](E13_operational_scalability_95k/) | Zero-shot operational and decision-space scalability | separates usable-storage occupancy, geometry, and total episode workload under the frozen E14+D12 stack | complete: 45/45 strict |
| [E14](E14_certification_scalability_95k/) | Certificate-budget and latency scalability | retains the accepted E14 cleanup+D12 stack while varying native-search budget across E13's main grid | complete: predeclared sweep plus post-hoc 32/64-node refinement |
| [E16](E16_prediction_ranking_audit/) | Prediction-to-ranking mechanism audit | connects candidate-relative future forecasts to exact ranking changes, paired continuations, action-type/residual selector effects, signed timing consequences, and a matched component rollout ablation | complete: E16-D 360/360 strict |
| [E19](E19_service_tails_cost_sensitivity/) | Service tails and operational-cost sensitivity | reanalyzes closed E1/E11/E13 records without training or new rollouts | complete: 1,860 E1 rows, 28 E11 coordinates, 15 E13 coordinates |

The [development history](development/) records D1--D12: architecture screens,
convergence checks, scalability support, shared-search diagnosis, and the
constructive relocation-family certification and its integrated confirmation.
Future paper experiments
use their declared identifiers independently of that development sequence.

E4 and E5 share one prospective 30-instance 92k panel. E5 reuses E4's full
conditioned rows as controls; it does not duplicate those rollouts. E5(a) is
assembled from completed diagnostics and adds no run. E5(b) is explicitly an
inference-time conditioning-input ablation, not a conditioned-training
ablation.

Generated checkpoints, ledgers, reports, tables, and figures remain under
ignored `results/` paths. The numbered launchers delegate to frozen protocol
directories whose paths are retained for result authentication.
