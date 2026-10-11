# Computational-cost reporting for Section 6

This report uses only existing frozen artifacts; it performs no training or evaluation. It distinguishes persisted measurements from unavailable quantities.

## Training-time audit

| Stage | Model seed | Recorded workload | Recorded timed workload | Complete elapsed training time | Training hardware |
|---|---:|---|---:|---|---|
| A: encoder + operational critic | 0 | 500 training episodes + 400 validation rollouts | 4.62 h* | unavailable | unavailable |
| A: encoder + operational critic | 1 | 500 training episodes + 400 validation rollouts | 5.82 h* | unavailable | unavailable |
| A: encoder + operational critic | 2 | 500 training episodes + 400 validation rollouts | 5.78 h* | unavailable | unavailable |
| B: handling predictor | 0 | 400 collection episodes; 8 rounds x 8 fit epochs | unavailable | unavailable | unavailable |
| B: handling predictor | 1 | 500 collection episodes; 10 rounds x 8 fit epochs | unavailable | unavailable | unavailable |
| B: handling predictor | 2 | 400 collection episodes; 8 rounds x 8 fit epochs | unavailable | unavailable | unavailable |

*Stage A's value is the sum of persisted in-episode wall clocks and the 400 scheduled validation-rollout clocks. It includes the agent updates executed inside those episodes, but excludes unmeasured process startup, checkpoint I/O, orchestration gaps, and any other untimed work. It is therefore an instrumented workload total, not complete elapsed training time. Stage B persisted neither collection nor fitting wall clocks. Neither stage persisted training-machine details. These unavailable values must not be inferred from episode or round counts.*

## Hardware and runtime contexts

| Measurement context | Device and machine | Thread settings | Software / platform | Availability |
|---|---|---|---|---|
| Stage A training | unavailable | unavailable | unavailable | Training artifacts did not persist machine metadata |
| Stage B training | unavailable | unavailable | unavailable | Training artifacts did not persist machine metadata |
| E1 common evaluation | cpu; CPU model unavailable | PyTorch threads=1 | Python 3.10.12; NumPy 2.2.6; PyTorch 2.5.1+cu124; platform unavailable | Runtime/software recorded; machine model unavailable |
| E13 latency measurement | Intel(R) Core(TM) Ultra 9 285; 24 logical CPUs; device=cpu | intra-op=24; inter-op=24; isolated process per episode | Python 3.10.12; PyTorch 2.5.1+cu124; Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35 | Fully recorded in every main-panel ledger |

Hardware contexts are not interchangeable. E13 decision-time measurements must be reported with the E13 machine and thread settings; the E1 one-thread contract describes benchmark execution but does not identify its processor. Current-machine specifications must not be substituted retrospectively for missing training or E1 hardware metadata.

## Frozen-controller inference and decision timing

Timing hardware: **Intel(R) Core(TM) Ultra 9 285**, 24 logical CPUs, CPU inference, PyTorch 2.5.1+cu124, 24 intra-op and 24 inter-op threads, Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35. Every episode ran in an isolated process. These settings apply to E13 latency only.

The following component distribution pools 894 warm decisions from the nine E13 size-by-occupancy coordinates. The first decision of every episode is excluded.

| Component | Median (ms) | p95 (ms) | Maximum (ms) |
|---|---:|---:|---:|
| Graph + candidate encoding | 304.90 | 949.69 | 1558.81 |
| Operational prediction Qop | 1.63 | 22.40 | 75.89 |
| Handling prediction QN | 1.06 | 30.97 | 103.22 |
| Prediction heads total (Qop + QN) | 3.50 | 53.75 | 135.46 |
| Neural inference total | 310.16 | 1003.93 | 1607.25 |
| Certified-frontier construction | 296.25 | 26194.08 | 32374.82 |
| Selector/guard residual | 7.00 | 55.00 | 238.92 |
| Complete controller decision | 995.83 | 26555.70 | 32745.78 |

`Neural inference total` is encoding + Qop + QN. `Selector/guard residual` is the measured selection time not captured by the neural hooks; it includes mode aggregation, ranking, the liveness guard, and bookkeeping. `Complete controller decision` is certified-frontier construction plus the full selection wall time.

### Warm end-to-end decision latency by scale and occupancy

| Coordinate | Warm decisions | Neural median (s) | Neural p95 (s) | Decision median (s) | Decision p95 (s) | Decision maximum (s) |
|---|---:|---:|---:|---:|---:|---:|
| 5x5 L (25.0%) | 43 | 0.223 | 0.769 | 0.238 | 0.785 | 0.973 |
| 5x5 M (50.0%) | 39 | 0.233 | 0.969 | 0.255 | 1.007 | 1.042 |
| 5x5 H (75.0%) | 64 | 0.263 | 0.967 | 0.279 | 0.990 | 1.125 |
| 8x8 L (31.4%) | 74 | 0.466 | 1.089 | 1.083 | 2.979 | 3.967 |
| 8x8 M (51.4%) | 103 | 0.206 | 0.864 | 0.560 | 3.349 | 3.644 |
| 8x8 H (71.4%) | 127 | 0.158 | 0.640 | 0.708 | 3.936 | 4.906 |
| 10x10 L (30.2%) | 106 | 0.299 | 0.769 | 1.024 | 15.411 | 16.774 |
| 10x10 M (50.8%) | 149 | 0.327 | 1.374 | 3.377 | 26.682 | 28.985 |
| 10x10 H (69.8%) | 189 | 0.367 | 1.123 | 8.999 | 30.566 | 32.746 |

These latency distributions describe the decisions actually reached at each coordinate. They are not simulation steps, delivery-time errors, or matched counterfactual timings across different scale/occupancy regimes. Certification includes candidate generation, exact search, cache handling, and D12 family-proof construction; the neural columns measure only the hooked forward-pass components.

## Recommended manuscript wording

Complete elapsed training time was not captured. Stage A logs retain per-episode and scheduled-validation wall clocks, whose sums range from 4.62 to 5.82 h per seed, but these are instrumented workload totals rather than process-wide training times. Stage B collection and fitting time was not persisted. Frozen-controller latency was measured separately on the stated CPU platform; neural forward-pass time is reported independently from certified-frontier construction and complete controller decision time.

Machine-readable report SHA-256: `0aa59fcd9485d2ecf44e58ac494b21a85a713a10c8dfad89d360a003008b51d2`.
