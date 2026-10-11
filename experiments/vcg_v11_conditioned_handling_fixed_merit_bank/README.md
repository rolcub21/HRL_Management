# Fixed-bank conditioned-handling diagnostic

This post-training diagnostic evaluates the final seed-0/1/2 conditioned
handling models on exactly the same small bank of training-probe states and
candidate frontiers. It records, for every candidate and lambda coordinate,
the frozen operational value, immediate handling term, predicted future
handling, total handling value, weighted handling penalty, and final merit.

It does not train, select a checkpoint, or open an evaluation panel. The bank
uses at most the first four frontiers from each of the nine predeclared
training probes. Duplicate frontiers are collapsed before model comparison.

Run:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/vcg_v11_conditioned_handling_fixed_merit_bank/run.sh
```

Outputs are written under
`results/vcg-v1-1-conditioned-handling-fixed-merit-bank/`.
