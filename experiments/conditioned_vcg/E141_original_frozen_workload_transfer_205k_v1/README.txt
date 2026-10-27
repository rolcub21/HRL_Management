E141 ORIGINAL SUCCESSFUL MODEL: SAME-LAYOUT WORKLOAD TRANSFER
===========================================================
NO TRAINING. Uses ORIGINAL residual Stage-A models: E131 seed0 and E133 seeds1/2,
all at fixed endpoint60. Not the newer mixture models. All checkpoints are frozen
as inference-only copies; original weights/target weights/config are hashed and
remain unchanged. Never select the best model/checkpoint on the transfer results.

    cd /home/ai_diagnosis/HRL_Management
    bash experiments/conditioned_vcg/E141_original_frozen_workload_transfer_205k_v1/run.sh run

One command runs native control first, then workload transfer, all with greedy
epsilon zero and no optimizer/replay changes. Long evaluation is user launched.
CPU one thread; no GPU inference required. Existing mixture training/evaluation
is not stopped or altered. CPU/memory contention can still change latency.

Optional split commands:
    bash experiments/conditioned_vcg/E141_original_frozen_workload_transfer_205k_v1/run.sh run --panel control
    bash experiments/conditioned_vcg/E141_original_frozen_workload_transfer_205k_v1/run.sh run --panel workload
    bash experiments/conditioned_vcg/E141_original_frozen_workload_transfer_205k_v1/run.sh status

Native control: exact earlier five development instances 187800..187804, empty
start, gap30, Poisson80, dates a+p, same8x8 geometry. Compare reconstructed costs
against each source checkpoint's episode60 historical validation, recorded in
native-control-reproduction.json. The historical late-window mean (48/54/60)
is NOT the exact endpoint60 number and should not be mistaken for reproduction.
Inspect discrepancies before attributing transfer failures to workload changes.

Workload transfer: the six already-fixed E138 development instances, same8x8
geometry, initial fill0/40/68.6%, calibrated offered loads. Dates constructed by
independent reference completion+slack; thus this changes due-date distribution
as well as arrivals/occupancy. It is NOT a clean one-factor arrival-gap ablation.
No new layout or yard size, no mixture fitting, no sealed test-panel opening.

All three original models and Dynamic PSLAP, GA, certified Dynamic use every
instance: 5*6=30 native-control rows, 6*6=36 workload rows, 66 total. Reuse saved
rows on rerun (including failures). Equal 900s wall cap,6000-step evaluation cap,
1000 decisions,2000 certificate nodes per row. Worst-case cap budget16.5h; most
simple baseline rows are short. Cap is empirical, not a uniform completion proof.
Original checkpoint3600 feature/time normalization is RETAINED, even though the
execution cap is6000; observations beyond native range may clip as before.
Only released external due dates feed the existing signed slack and nominal K.
Original timing scale, network weights, graph architecture, grouped selection,
outflow certificate, guard and bounded-wait restrictions are not redesigned.

Baselines receive the same instances/dates/accounting. Native baseline compound
retrieval versus VCG elementary recovery remains a comparison limitation. This
panel reuses development data, so there are no independent-test or significance
claims. Original and mixture outcomes stay separately labelled. A later new-layout
test or adaptation study is a different experiment, not silently mixed in here.

Output reuses the E140 recorder/format: e140-table.md and e140-report.json in this
E141 results directory, plus native-control-reproduction.json, original-model
provenance, per-row job/action traces. Control indices1..5; workload indices6..11.
Completion first, incomplete costs not ranked, J/components,J_W,timing,wait,
rehandles,work/steps and behavior traces retained. Ctrl-C/SIGTERM saves the current
partial row and exits. No original timeout, model or experiment is overwritten.
