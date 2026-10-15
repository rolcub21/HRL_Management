# Exact-v6 one-episode result

The isolated episode-26 run completed successfully from the authenticated
episode-25 snapshot (`f4695ba0...`) on the second RTX A5000. It did not modify
or commit a continuation checkpoint.

| Measure | v5 live episode 26 | v6 isolated episode 26 |
|---|---:|---:|
| Wall time | 1,543.58 s | 411.67 s |
| Learning time | 1,373.93 s | 256.90 s |
| Deliveries | 64/64 | 64/64 |
| Macro decisions | 156 | 151 |
| Dense rescored return | 2,237.61 | 2,352.61 |
| Mean absolute timing error | 38.50 | 37.50 |

This is a 3.75× end-to-end wall-time improvement and a 5.35× reduction in
instrumented learning time for this paired episode seed. It is one episode,
not a throughput guarantee.

V6 is mathematically equivalent to v5 within the tested tensor/Q/TD numerical
tolerances, but it is not bitwise trajectory-equivalent on CUDA. Small score
roundoff differences can change a near-tied action, after which the training
trajectory diverges. The differing decision counts and returns above are
evidence of that limitation.

At episode completion the process-local graph cache held 726,861 entries,
reported 83.10% hits and no evictions, and retained 4,361,166,000 bytes of
feature-tensor payload (about 4.06 GiB), excluding Python-object overhead.
This memory cost must be considered before adopting the one-million-entry
bound for longer runs.

Evidence:

- `results/e79-v6-exact-episode26-attempt2/report.json`
- live v5 comparison:
  `/home/bukuj/hrl-eval/results/vcg-e79-direct-legal-overdue-progress-full-10x10-v1/episode-reports/episode-0026.json`
- the failed first launch contains no episode work; it exposed and led to a
  tested v5-compatible TD-function alias.

## Terminal continuation

The authenticated v6 continuation completed episodes 29--63 and reached the
matched 4,032-job exposure at episode 63.  The overnight segment from episode
31 through 63 took 4.06 hours, averaging 7.38 minutes per episode.  Every
recorded episode delivered all 64 jobs.  The terminal episode reported:

| Measure | Episode 63 |
|---|---:|
| Dense rescored return | 2,474.31 |
| Deliveries | 64/64 |
| Macro decisions | 160 |
| Mean absolute timing error | 36.625 |
| Wall time | 8.3 min |

Terminal checkpoint SHA-256:
`d16c45ac18f56d6c49e893ec49415d6dc8cd5a602f390bc0e5482d663414c390`.
The checkpoint includes optimizer and replay state and is 1,006,418,362 bytes;
it is distributed as a release asset rather than a normal Git blob.

## Terminal fixed-development evaluation

The frozen episode-63 policy was evaluated greedily on the five prespecified
development seeds.  Evaluation performed no replay writes, gradient steps, or
checkpoint updates, and the source checkpoint hash was unchanged afterward.

| Seed | Delivered | Dense return | MAE | Mean signed error | Time |
|---:|---:|---:|---:|---:|---:|
| 112100 | 64/64 | 2,407.87 | 37.016 | -37.016 | 3.3 min |
| 112101 | 64/64 | 2,327.47 | 37.594 | -37.594 | 3.3 min |
| 112102 | 64/64 | 2,667.75 | 34.859 | -34.859 | 2.8 min |
| 112103 | 64/64 | 3,019.89 | 31.969 | -31.906 | 3.1 min |
| 112104 | 64/64 | 2,824.27 | 33.609 | -33.547 | 3.1 min |

Aggregate results:

- physical and operational completion: 100% (320/320 deliveries);
- mean dense rescored return: 2,649.45;
- mean absolute timing error: 35.009 steps;
- mean signed timing error: -34.984 steps (strong early-delivery bias);
- within the +/-20 target window: 12.8125%;
- tardy-delivery rate: 0.625%;
- relocations: 312, or 97.5 per 100 deliveries;
- action counts: 320 Accept, 320 Deliver, 312 Reconfigure, and 0 Defer.

The result supports reliable operational completion on this development panel,
but not good timing control.  In particular, the zero Defer count explains the
dominant early-delivery error and motivates tracing Deliver-versus-Defer mode
values before further training.  This uncertified protocol is not deployment,
safety, sealed-test, or paper evidence.

Canonical compact evidence is under
`results/vcg-e79-v6-direct-legal-overdue-progress-full-10x10-v1/`.
