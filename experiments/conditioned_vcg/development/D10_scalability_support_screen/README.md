# D10 — Scalability support and instrumentation screen

D10 is the development gate for Block 6. It checks which scales the existing
simulator, exact verifier, and frozen final VCG can execute before the formal
E13/E14 panels are frozen. It performs no training and does not read or modify
the running E12 representation experiment.

The follow-up [occupancy and 10×10 extension](OCCUPANCY_EXTENSION.md) measures
actual usable-storage occupancy and separates it from total episode workload.
It is a new authenticated protocol and leaves the completed v2 artifacts
unchanged.

Protocol v2 uses a workload-size-aware timing reducer. The superseded v1
output is retained as development history; it could summarize only eight-job
rows and therefore stopped before writing the 12-job result.

The graph controller is structurally variable-size compatible: node
coordinates and action locations are normalized, graph pooling accepts a
variable number of cells, and neither operational nor handling heads have a
fixed 5×5 input width. D10 therefore tests zero-shot transfer first.

## Factor-isolated scale screen

| ID | Axis | Yard | Blocks | Capacity | Workload/capacity |
|---|---|---:|---:|---:|---:|
| `reference_5x5_n8` | trained reference | 5×5 | 8 | 8 | 1.000 |
| `geometry_6x6_n8` | geometry only | 6×6 | 8 | 15 | 0.533 |
| `geometry_7x7_n8` | geometry only | 7×7 | 8 | 24 | 0.333 |
| `geometry_8x8_n8` | geometry only | 8×8 | 8 | 35 | 0.229 |
| `workload_6x6_n12` | workload at 6×6 | 6×6 | 12 | 15 | 0.800 |
| `workload_6x6_n15` | workload at 6×6 | 6×6 | 15 | 15 | 1.000 |
| `coupled_7x7_n18` | coupled size/workload | 7×7 | 18 | 24 | 0.750 |
| `coupled_8x8_n26` | coupled size/workload | 8×8 | 26 | 35 | 0.743 |

Every geometry uses one entrance/pickup and exactly three exits. Arrival rate
10 and Poisson dwell mean 80 remain fixed. The screen uses final model seed 0,
`lambda=.10`, the deployed 20,000-node certification cap, and development seed
95000 first. These choices diagnose support and runtime; they are not a paper
performance comparison.

## Recorded quantities

For every decision D10 records physical and certified frontier sizes by action
type; SAFE/UNSAFE/UNKNOWN counts; exact-search node expansions and latency;
candidate-generation overhead; graph/action representation time; operational
and handling-head scoring time; selection residual; complete decision latency;
simulated steps; liveness interventions; memory observations; and exceedance
of predeclared 0.1, 1, and 5 second thresholds. UNKNOWN remains rejected and no
deadline fallback is added.

Certification-call expansion distributions cover actual cache misses. Overall
candidate UNKNOWN rates use the frontier audit over every candidate that
received a SAFE, UNSAFE, or UNKNOWN certificate, including cached results;
Defer is excluded because it is not a recoverability-search subject.

## Run order

Prepare now without competing with the running E12 training:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/development/D10_scalability_support_screen/run.sh prepare
```

When compute is available, start with only three small coordinates:

```bash
bash experiments/conditioned_vcg/development/D10_scalability_support_screen/run.sh run-small
```

Then evaluate one larger coordinate at a time, or all eight seed-95000 pilot
coordinates:

```bash
bash experiments/conditioned_vcg/development/D10_scalability_support_screen/run.sh run-scale coupled_7x7_n18
bash experiments/conditioned_vcg/development/D10_scalability_support_screen/run.sh run-pilot
```

The completed v2 screen below establishes support through 8×8. An occupancy-
controlled 10×10 extension is still required before the formal E13 scale
ladder and E14 verifier budgets are frozen.

## Gate outcome

The three-instance screen completed all 24 rows strictly, with no UNKNOWN
certificate and no empty certified frontier. This establishes implementation
support, not expected operational performance.
The timings are descriptive development measurements; E14 must repeat its
hardware audit under an idle, declared execution environment.

| Scale | Strict | Candidates/state | Median decision | p95 decision | MAE | Rehandles/100 |
|---|---:|---:|---:|---:|---:|---:|
| 5×5, 8 blocks | 3/3 | 17.7 | 0.80 s | 0.94 s | 11.67 | 8.33 |
| 6×6, 8 blocks | 3/3 | 46.2 | 1.11 s | 1.39 s | 10.58 | 12.50 |
| 7×7, 8 blocks | 3/3 | 66.3 | 1.27 s | 1.98 s | 16.38 | 50.00 |
| 8×8, 8 blocks | 3/3 | 46.4 | 1.05 s | 1.28 s | 65.21 | 0.00 |
| 6×6, 12 blocks | 3/3 | 38.2 | 0.99 s | 1.65 s | 37.75 | 30.56 |
| 6×6, 15 blocks | 3/3 | 36.0 | 0.94 s | 1.52 s | 35.51 | 22.22 |
| 7×7, 18 blocks | 3/3 | 87.6 | 1.92 s | 7.95 s | 134.81 | 79.63 |
| 8×8, 26 blocks | 3/3 | 50.2 | 1.16 s | 1.67 s | 71.33 | 0.00 |

The 7×7/18 coordinate, rather than the nominally largest coordinate, was the
computational stress case across all three instances: mean frontier
construction reached 2.36 seconds and p95 end-to-end latency reached 7.95
seconds. The 8×8/26 trajectories chose no reconfiguration, kept certificates
shallow, and were consequently cheaper. Runtime is therefore state- and
policy-trajectory-dependent.

The v2 ratios are explicitly total arrivals divided by usable storage
capacity; they are not state occupancy. The screen does not yet measure
initial, time-weighted mean, or peak occupancy. Accordingly, `5×5/8`,
`6×6/12`, `7×7/18`, and `8×8/26` remain a provisional coupled ladder, and the
budget grid `{2, 4, 8, 16, 20000}` remains provisional too. The complete v2
expansion range was 1--42 nodes. The occupancy-controlled 10×10 extension must
test low, medium, and high realized occupancy before either formal design is
sealed.
