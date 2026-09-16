# D10 occupancy and 10×10 extension

This companion protocol corrects the limitation of D10 v2: total jobs divided
by capacity is not state occupancy. It leaves the authenticated v2 source and
results unchanged.

Each run starts from a declared warm yard containing `K` stored blocks. Actual
storage occupancy is measured against
`len(env.storage_positions)` initially and after every primitive simulator
step. Concurrent admitted workload, physical presence, arrived-but-unadmitted
work, and total jobs are separate fields.

## Main size-by-occupancy grid

| Yard | Usable slots | Low | Medium | High |
|---|---:|---:|---:|---:|
| 5×5 | 8 | 2/8 = 25.0% | 4/8 = 50.0% | 6/8 = 75.0% |
| 8×8 | 35 | 11/35 = 31.4% | 18/35 = 51.4% | 25/35 = 71.4% |
| 10×10 | 63 | 19/63 = 30.2% | 32/63 = 50.8% | 44/63 = 69.8% |

Every main coordinate adds two later arrivals so admission remains represented.
The placement is deterministic, unique, avoids the robot start, and fills from
left to right to retain a right-hand recovery corridor.

Companion coordinates add:

- fixed concurrent workload `K=6, N=8` across 5×5, 8×8, and 10×10 (the 5×5
  coordinate is the main high-occupancy row);
- fixed 8×8 medium initial occupancy with total episode lengths 20, 36, and 54;
- matched-capacity 6×10 and 10×6 medium-occupancy rectangles.

This remains a development support screen. A failure is classified separately
as an implementation/unsupported error, certificate-budget limitation, or
valid operational incompletion. UNKNOWN is always rejected and is never
reported as proof of unrecoverability.

The completed 10×10 low-occupancy pilot exposed multi-minute tail latency.
The separate [latency diagnostic](LATENCY_DIAGNOSTIC.md) runs only the medium
and high 10×10 coordinates with per-decision/per-certificate attribution and a
predeclared censoring limit; it does not alter this authenticated protocol.

## Staged execution

Prepare the immutable instances and contract:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/development/D10_scalability_support_screen/run_occupancy.sh prepare
```

Start with the three 10×10 coordinates for development seed 95100:

```bash
bash experiments/conditioned_vcg/development/D10_scalability_support_screen/run_occupancy.sh run-core-pilot
```

The commands are resumable. Run one coordinate when a dense case needs to be
inspected separately:

```bash
bash experiments/conditioned_vcg/development/D10_scalability_support_screen/run_occupancy.sh \
  run-scenario size_10x10_occ_high 1
```

`run-grid-pilot` covers the nine main coordinates for one seed. `run-pilot`
covers all 15 coordinates for one seed. `run-grid` runs the three development
seeds on the nine main coordinates; `run-all` runs all 45 rows. Do not launch
the complete screen until the staged 10×10 cases establish approximate cost.
