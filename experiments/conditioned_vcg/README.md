# Conditioned VCG experiments

Paper-facing experiments use zero-padded `E` identifiers. Development screens
use a separate `D` sequence so rejected architectures are not confused with
the experiments that support the final paper.

| ID | Experiment | Role | Status |
|---|---|---|---|
| [E1](E01_benchmark_90k/) | Prospective common-panel benchmark | primary timing--handling result | complete: 1,860/1,860 rows |
| E2 | Generalization across yard/workload settings | reserved | planned |
| [E3](E03_certification_ablation_90k/) | Candidate-source certification ablation | tests whether the verifier prevents self-blocking | complete: 90/90 certified, 86/90 physical-only |
| [E4](E04_safe_frontier_ranking_92k/) | Certified-frontier ranking ablation | isolates management quality added by the ranking signal | prepared program |
| [E5](E05_handling_model_ablation_92k/) | Handling validation and conditioning-input ablation | validates `Q_N` and tests its deployment-time preference dependence | prepared program |
| [E11](E11_distribution_shift_93k/) | Frozen-policy distribution-shift generalization | tests completion and timing--handling quality under workload and mirrored-access shifts | prepared program |
| [E12](E12_representation_ablation_94k/) | Representation attribution under shift | independently retrains relational, nonrelational, and successor-free controllers | prepared program |
| E13 | Zero-shot operational and decision-space scalability | separates usable-storage occupancy, geometry, and total episode workload | awaiting staged D10 occupancy/10×10 results |
| [E14](E14_certification_scalability_95k/) | Exact-certificate reuse and latency scalability | separates timing-key reuse, path cleanup, and constructive witness reuse on ordered 10×10 query workloads | D12 accepted: strict medium/high completion, 21,051 relocation-family proofs, zero misses |

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
