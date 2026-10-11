# Seed-1 convergence-controlled continuation

Seed 1 completed the matched two-phase recipe but did not pass the training
convergence gate. This continuation starts from that immutable round-8
terminal and retains the finalized controller, data construction, fitting
settings, and `rho=0.25` parameter damping.

The stopping rule is predeclared:

- stop at the first two consecutive stable training-probe transitions;
- run at least two and at most four additional rounds;
- if the four-round cap is reached without convergence, do not evaluate;
- never use evaluation performance or checkpoint selection for stopping.

Each round contains 50 collection episodes and nine fixed training probes.
Thus the continuation runs between 118 and 236 simulator episodes. Building
the common fixed diagnostic bank adds nine one-time training-probe rollouts.

The fixed bank records `Qop`, immediate and future `QN`, `lambda*QN`, final
merit, and selected candidate at the parent terminal and every new round. It
is diagnostic only and cannot affect fitting or stopping.

Run the complete resumable workflow:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/vcg_v11_conditioned_handling_seed1_convergence_continuation/run.sh run-all
```

The script resumes automatically from the latest completed round after an
interruption.
