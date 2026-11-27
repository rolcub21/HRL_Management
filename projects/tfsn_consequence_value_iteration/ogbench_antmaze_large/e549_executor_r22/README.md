# R22: E549-style executor on OGBench AntMaze Large

R21 rejected the official 1M CRL checkpoint: 18% of adaptive-reach chains
inverted, against the predeclared 5% ceiling. R19 and R20 remain preserved as
failed CRL fine-tuning attempts. R22 is a separate from-scratch run.

Prepare and verify the frozen data and E549 comparison:

```bash
cd /home/ai_diagnosis/HRL_Management
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh verify
```

Train and audit the first gate:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh train-1m
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh audit-1m
```

Only if 1M fails, continue the same checkpoint and replay buffer:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh resume-2m
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh audit-2m
```

Only if 2M also fails:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh resume-5m
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh audit-5m
```

The runner refuses a continuation after a passing checkpoint. Acceptance
requires at most 5% inverted R21 chains and at least the official checkpoint's
38.1% R21 contact rate, preventing immobility from passing as safety.
