# E121: frozen blocked-pickup waiting

Separate controller ablation following E120's exact-parity attribution. E119's
due+20 change is not adopted. The source E118 model and ongoing training are
untouched. No weights are fitted here.

Only when an inbound job exists and the physical Accept mask is empty, offer
Defer subject to all other native checks: current SAFE recovery, positive
nominal stored due slack, 10-step macro cap, 16-Defer streak limit. Additionally
stop offering inbound waiting once that job's age since release reaches 80
steps. This is an eligibility bound, **not a guaranteed 80-step admission SLA**.
The guard and exact certificate gate remain active. No new formal liveness
theorem is claimed for this separately changed controller.

```bash
bash experiments/conditioned_vcg/E121_blocked_pickup_wait_8x8_185k_v1/run.sh run-panel
```

Requires completed E120 and its E119 repair1 source artifacts. Runs five reused
development instances with epsilon zero; CPU, one thread, 300-second episode
bound. Reports paired objective cost and service/early-tail/completion metrics
against the existing E119 native rows, plus how often the newly available wait
is actually selected. A negative result does not rule out retraining: the
frozen critic has not been trained on inbound Defer choices. Conversely,
completion on this panel is not a general safety or liveness proof.
