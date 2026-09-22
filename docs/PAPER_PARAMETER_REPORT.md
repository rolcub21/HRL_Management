# Paper parameter report

This report records the parameters that governed the final VCG controller and every comparator in the common 90k E1 benchmark, followed by the coordinates of the paper's principal confirmation experiments. Values were reconstructed from sealed contracts, exact selected checkpoints, and authenticated run ledgers; no policy was rerun. Development-only screens and superseded contracts are outside this report.

## Shared problem and evaluation settings

| Parameter | Value |
|---|---|
| Grid | 5 x 5 |
| Usable storage cells | 8 |
| Start / pickup / waiting | [1, 1] / [1, 3] / [0, 3] |
| Exit cells | (4, 1), (4, 2), (4, 3) |
| Blocks / required deliveries | 8 |
| Arrivals | exponential, rate=10, mean inter-arrival=0.1 step; first arrival at step 0 |
| Storage durations | Poisson, mean=80 steps |
| Episode cap | 2000 primitive steps |
| Evaluation panel | 30 frozen EpisodeInstances, seeds 90000--90029 |
| Target window | +/-20 primitive steps |
| Execution | CPU, one PyTorch thread; Python 3.10.12, NumPy 2.2.6, PyTorch 2.5.1+cu124 |

All methods used the same serialized EpisodeInstances. The statistical unit was the EpisodeInstance; model, policy, and optimizer repetitions were averaged within instance. Evaluation performed no learning, checkpoint selection, or preference selection, and complete-case filtering was forbidden.

### Reward and reported objectives

For delivery error `e`, the dense delivery reward was `40 - 1.5|e| - 0.5 max(0, |e|-20)`. Every primitive step contributed `-0.09`, and the first successful storage of a block contributed `+5`. Thus episode return was not used as a substitute for either MAE or physical rehandles; those metrics were reported separately.

| Parameter | Value |
|---|---|
| Primary coordinates | MAE; physical rehandles per 100 required deliveries |
| Secondary metrics | dense return; primitive steps; within-window rate |
| Rehandle rate | 100 x storage-to-storage relocations / 8 required deliveries |
| Failure handling | retain failures; suppress a method/coordinate aggregate rather than complete-case average |

## Final VCG

The deployed score was `Q_op(s,c) - lambda [1{Reconfigure} + Q_N,future(s,c,lambda)]`. At `lambda=0`, the implementation bypassed the handling head and delegated exactly to the frozen operational controller.

### Operational critic and selector

| Parameter | Value |
|---|---|
| Representation | shared counterfactual graph; 15 node and 28 action features |
| Graph / embedding / action / head widths | 64 / 64 / 32 / 128 |
| Message-passing steps | 3 |
| Discount / reward scale | 0.99 / 0.01 |
| Optimizer | Adam, lr=5e-05, batch=128, replay=20000 |
| Updates | every macro; target every 200 updates; Huber delta=1; gradient clip=5 |
| Training | 500 episodes per model seed; epsilon 0.90 to 0.05 over 3,000 macro decisions |
| Validation / selection | every 25 episodes on seeds 79000--79019; selected episodes seed 0/1/2 = 500/450/375 |
| Modes | Accept, Recover, Defer; Recover contains Deliver and Reconfigure |
| Temperatures | within-mode 0.1; mode 1.0; deterministic regularized mode argmax then candidate argmax at deployment |

### Exact certificate and liveness

| Parameter | Value |
|---|---|
| Search | goal-directed; max nodes=20000; depth and primitive-step bounds unbounded |
| Queue cells | reserved in recovery search |
| UNKNOWN / UNSAFE | excluded (fail closed) |
| Defer limits | 10 option steps and 16 consecutive defer decisions |
| Nonprogress recovery limit | 2 |

### Conditioned future-handling head

| Parameter | Value |
|---|---|
| Input / hidden layers | 224 frozen candidate features plus normalized lambda; two 128-unit SiLU layers |
| Output | softplus nonnegative future physical rehandles |
| Target | undiscounted post-current-macro Monte Carlo rehandles from complete episodes |
| Preference range | [0, 0.2] during collection; fixed within each episode |
| Collection episodes | seed 0/1/2 = 400/500/400 in 50-episode rounds |
| Fit | 8 epochs/round, batch 256, lr=5e-4; first 4 full-update rounds, then Polyak rho=0.25 |
| Operational critic | frozen throughout handling-head training |
| E1 lambda grid | 0, 0.025, 0.0375, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2 |

## Baseline parameters

### Historical VCG 2.3

| Parameter | Value |
|---|---|
| Architecture | shared graph trunk with operational and nonnegative rehandle heads; widths 64/64/32/128; 3 message-passing steps |
| Discounts | operational=1.0; rehandle=1.0; operational reward scale=0.01 |
| Optimizer | lr=5e-05, batch=16, replay=2000, target every 50 updates |
| Training | 200 episodes per seed; 20-episode critic warm-up, then 10-episode dual blocks |
| Dual | budget=20 rehandles/100 deliveries; lr=0.01; lambda initial/max=0/20 |
| Selected seed operating points | seed 11: ep120, lambda=.152; seed 12: ep160, lambda=.170; seed 13: ep100, lambda=.213 |
| Deployment | stochastic induced nested Lagrangian policy; within/group temperatures=.01/.05 |
| Certificate | goal-directed, max 100,000 nodes, fail closed |

### Duration-aware Dynamic PSLAP

No parameters were learned. Inbound assignment retained only reachable empty cells that preserved access to earlier-due blocks, then minimized exit distance, travel distance, and cell index. Relocation minimized travel distance, exit distance, and cell index. Retrieval was dispatched when the complete travel/handling plan had nonpositive slack, ordered by slack, relocation count, plan length, and block label. The baseline was deterministic and did not use the exact VCG certificate.

### Capacity-aware rolling GA

| Parameter | Value |
|---|---|
| Information | stored inventory plus arrived pending blocks only |
| Population / generations | 16 / 10 at every inbound decision |
| Elites / tournament | 2 / 3 |
| Crossover | one-point with uniqueness repair; probability=0.8 |
| Mutation probability | 0.1 |
| Fitness | lexicographic infeasible known placements, predicted obstruction, route length |
| Replications | 4 optimizer RNG realizations per EpisodeInstance; never best-of-4 |

### A3C adaptation

| Parameter | Value |
|---|---|
| Status | Kim et al. (2020)-inspired adaptation; not an exact reproduction |
| Observation | 5 channels: relative dwell, occupancy, storage mask, traversability, exit proximity |
| Network | two 3x3 convolution layers with 32 channels; masked per-cell actor and scalar critic |
| Optimizer | Adam, lr=0.0001, weight decay=0 |
| Loss / update | gamma=0.99; entropy=0.01; value=0.5; gradient clip=5; 1 update/eligible episode |
| Training reward | negative relocations attributed to the placed block; incomplete/censored episodes skipped |
| Training / validation | 1,000 episodes per seed; validation every 100 episodes on five fixed instances with five stochastic rollouts |
| Selected episodes | seed 0/1/2 = 100/200/700 |
| Deployment | stochastic masked categorical placement; 5 policy RNGs per model-instance |
| Physical control | common deterministic navigation and atomic storage/retrieval executors |

## Raw run accounting

| Parameter | Value |
|---|---|
| VCG | 900 = 30 instances x 3 model seeds x 10 lambda values |
| Historical VCG 2.3 | 360 = 30 x 3 model seeds x 4 policy RNGs |
| Dynamic PSLAP | 30 = 30 x one deterministic run |
| Capacity-aware GA | 120 = 30 x 4 optimizer RNGs |
| A3C adaptation | 450 = 30 x 3 model seeds x 5 policy RNGs |

These unequal raw counts are intentional nuisance replication. They do not represent different test panels: every method was evaluated on the same 30 instances, and inference was performed at the instance level after within-instance averaging.

## Principal experiment coordinates

These settings are experimental coordinates, not additional learned hyperparameters. Reused controls are identified so that reanalysis is not mistaken for independent replication.

| Parameter | Value |
|---|---|
| E3: certification | physical-feasibility-only vs recoverability-certified; lambda=.10; 3 model seeds x 30 instances; 90 new physical-only rows, certified control reused from E1 |
| E4: ranking | random-safe, heuristic-safe, frozen Qop-safe, and conditioned VCG at lambda in {.05,.10,.20}; 3 model seeds x 30 instances; five random-ranking seeds |
| E5(a): model validation | 34 fixed states, 686 candidates, 17 lambda values |
| E5(b): preference input | deployment lambda in {.05,.10,.20}; comparator clamps predictor input to .10 while retaining requested deployment multiplier |
| E5(c): future consequences | immediate Reconfigure cost only vs fixed learned future-handling estimate; lambda in {.05,.10,.20}; 270 new immediate-only rows, controls reused from E4/E5(b) |
| E8: liveness | descriptive instrumented pilot; 8x8 high occupancy, instance 95100, model seed 0, lambda=.10, 20,000-node exact-search budget and 600-s wall limit |
| E16: score component | immediate-only, action-type-only, candidate-residual-only, full-future; predictor and deployment lambda=.10; 3 model seeds x 30 instances = 360 rows |
| E19: service cost | analysis only; tardiness weights {1,2,4}, rehandle equivalents {0,10,25,50} timing steps, job-level quantiles {.50,.90,.95,.99} |

### E11 distribution-shift panel

All E11 policies were frozen. The panel used model seeds 0--2 and instance seeds 93000--93029. Frozen Qop represented lambda=0; conditioned VCG used lambda in {.05,.10,.20}; the heuristic-safe comparator was deterministic. No deployment-specific preference selection was allowed.

| Regime | Shift | Mean inter-arrival | Duration | Entry |
|---|---|---:|---|---|
| reference | in_distribution_reference | 0.1 | poisson (mean 80) | reference |
| arrival_spread | arrival_intensity | 10 | poisson (mean 80) | reference |
| dwell_short | storage_duration | 0.1 | poisson (mean 40) | reference |
| dwell_long | storage_duration | 0.1 | poisson (mean 120) | reference |
| dwell_bimodal | workload_composition | 0.1 | bimodal_40_120 (mean 80) | reference |
| mirrored_entry | entrance_geometry | 0.1 | poisson (mean 80) | mirrored |
| combined_shift | combined | 10 | bimodal_40_120 (mean 80) | mirrored |

The compared methods were frozen Qop-safe, conditioned VCG, and heuristic-safe. The complete design contained 2,730 rows; the exact same EpisodeInstance was shared within each regime-policy comparison.

### E12 representation ablation

| Parameter | Value |
|---|---|
| Representations | full relational successor; nonrelational successor; relational current-plus-candidate without explicit successor |
| Regimes | the seven E11 regimes |
| Preferences | lambda in {0,.05,.10,.20} |
| Seeds and rows | 3 model seeds x 30 instances; 7,560 corrected-executor rows |
| Training | 18 runs total: 9 operational runs (3 representations x 3 seeds, 500 episodes) and 9 handling runs (400 episodes = 8 rounds x 50); operational critic frozen during handling training |
| Corrected-executor confirmation | no retraining; 7,560 fresh evaluations of the frozen representation variants |

### E13 scale, occupancy, geometry, and episode length

E13 used frozen model seed 0 at lambda=.10 on instance seeds 95100--95102. The verifier used goal-directed exact search with a 20,000-node cap, queue-cell reservation, timing-invariant caching, order-preserving path cleanup, and validated relocation-family certificates. Runs were isolated by process with a 1,800-s episode wall limit; latency thresholds were .1, 1, and 5 s.

| Scenario | Comparison | Grid | Initial occupancy | Total jobs | Step cap |
|---|---|---:|---:|---:|---:|
| size_5x5_occ_low | size_by_occupancy | 5x5 | 2/8 (25.0%) | 4 | 2000 |
| size_5x5_occ_medium | size_by_occupancy | 5x5 | 4/8 (50.0%) | 6 | 2000 |
| size_5x5_occ_high | size_by_occupancy | 5x5 | 6/8 (75.0%) | 8 | 2000 |
| size_8x8_occ_low | size_by_occupancy | 8x8 | 11/35 (31.4%) | 13 | 2000 |
| size_8x8_occ_medium | size_by_occupancy | 8x8 | 18/35 (51.4%) | 20 | 2000 |
| size_8x8_occ_high | size_by_occupancy | 8x8 | 25/35 (71.4%) | 27 | 2700 |
| size_10x10_occ_low | size_by_occupancy | 10x10 | 19/63 (30.2%) | 21 | 2100 |
| size_10x10_occ_medium | size_by_occupancy | 10x10 | 32/63 (50.8%) | 34 | 3400 |
| size_10x10_occ_high | size_by_occupancy | 10x10 | 44/63 (69.8%) | 46 | 4600 |
| fixed_workload_8x8_k6_n8 | geometry_fixed_workload | 8x8 | 6/35 (17.1%) | 8 | 2000 |
| fixed_workload_10x10_k6_n8 | geometry_fixed_workload | 10x10 | 6/63 (9.5%) | 8 | 2000 |
| episode_8x8_medium_n36 | episode_length | 8x8 | 18/35 (51.4%) | 36 | 3600 |
| episode_8x8_medium_n54 | episode_length | 8x8 | 18/35 (51.4%) | 54 | 5400 |
| rectangle_6x10_medium | aspect_ratio | 6x10 | 16/31 (51.6%) | 18 | 2000 |
| rectangle_10x6_medium | aspect_ratio | 10x6 | 16/31 (51.6%) | 18 | 2000 |

### E14 certification-budget sensitivity

| Parameter | Value |
|---|---|
| Predeclared max-node budgets | 2, 4, 8, 16, and 20,000; the 20,000-node reference reused E13 |
| Post-hoc threshold refinement | 32 and 64 nodes, explicitly labeled post hoc |
| Panel | nine 5x5/8x8/10x10 low/medium/high scenarios; instance 95100, model seed 0, lambda=.10 |
| Proof isolation | fresh per-episode cache; anchors acquired under the arm's own budget; no cross-budget positive-proof reuse; UNKNOWN never reused as UNSAFE |
| Timing | isolated CPU process; .1/1/5-s decision thresholds; 1,800-s episode wall limit |

## Provenance

Machine-readable registry SHA-256: `9b07c47767f977e0c2caad4c2716f4025b8eed800034c3c0d3d4f8878d29741d`.

The companion `e1-parameter-register.json` contains full checkpoint digests and source-artifact hashes. Historical VCG 2.3 is reported separately because it is not a preference coordinate of the final conditioned VCG. The Kim2020-inspired comparator is named **A3C adaptation** throughout the paper.
