# Frozen TF-SN evaluation panel

Frozen on 2026-11-27, before evaluation. The machine-readable source of this
decision is
[`OGBENCH_EVALUATION_PANEL_FREEZE.json`](OGBENCH_EVALUATION_PANEL_FREEZE.json).
This file freezes the comparison matrix, not the exact run protocols. Every
runner must still freeze its exact OGBench environment ID, task rows, seeds,
budget, checkpoints, and estimands before accessing evaluation tasks.

| Comparison | Target | AntMaze large | AntMaze giant | HumanoidMaze medium | FetchPush |
| --- | --- | --- | --- | --- | --- |
| Hidden goal: TF-SN vs MEGA | TF-SN clearly better | run | run conditionally | run | run |
| Supplied goal: TF-SN vs SGM and reported HIQL | better than SGM; competitive with HIQL | run / reported | run conditionally / reported | run / reported | not applicable |
| Ablation: direct executor | isolate what planning adds | run | run conditionally | run | run |

## Fixed comparison decisions

- Run SGM, not SoRB. SGM is the closer edge-correcting comparison; running
  both would add training cost without answering a distinct central question.
- Use HIQL as the one reported maze representative. Its numbers must always be
  labelled **reported**, never as matched reruns. CRL and QRL are not trained
  separately for this panel.
- TF-SN, SGM, and the direct-executor ablation use the same executor and the
  same task rows within a domain.
- AntMaze giant is gated by AntMaze large. It is not started speculatively or
  in parallel; its exact success gate belongs in the AntMaze protocol.
- FetchPush is intentionally retained as a non-OGBench transfer column. The
  campaign is OGBench-centered, not exclusively OGBench.

## Stopped work

The Gymnasium PointMaze/AntMaze aligned multi-seed rebuild (`r16`) is stopped
and is no longer part of the main evaluation path. Do not continue its seed-0
evaluations or its seed-1/seed-2 preparation runs. Completed artifacts remain
development history until an explicit cleanup; they are not part of this
panel.

This stop does not redefine TF-SN. The frozen R74 specification remains the
method reference; only the benchmark and comparison campaign changed.

## Required reporting separation

Hidden-goal results, supplied-goal results, and the direct-executor ablation
are reported separately. OGBench matched runs must not be pooled with the old
Gymnasium maze rebuild, and published HIQL values must remain visibly distinct
from systems run under the new matched protocol.
