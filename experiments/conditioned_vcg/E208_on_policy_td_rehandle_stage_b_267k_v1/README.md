# E208 — on-policy TD future-rehandle Stage B

E208 replaces E204's roundwise Monte Carlo labels from earlier controllers with
one-step on-policy SARSA evaluation of the controller currently using the head:

```text
N_alpha(s,c) = E[I(c_next) + N_alpha(s_next,c_next)]
```

The next action is the action actually chosen by the same fixed-alpha episode
policy. The exact current rehandle remains outside the head. E191, certification,
the progress guard, macro execution, and all validated accelerations stay frozen.

The seed-0 pilot uses 36 episodes, with alpha sampled continuously and
stratified over `[10,20)`, `[20,30)`, and `[30,40)`. Every workload/alpha-band
pair receives four episodes. The fresh near-zero head initially implements the
stable immediate-only controller. A delayed target head is Polyak-updated after
each complete episode; stale Monte Carlo labels and replay are not used.

After training, nine endpoint episodes use the exact E205 development days at
lambda 10, 20, and 40. They report paired performance against E207, held-out
Bellman/Monte-Carlo errors, and the cross-price regret matrix. This is a pilot,
not multiseed paper evidence.

```bash
cd /home/ai_diagnosis/HRL_Management
set -o pipefail

MPLCONFIGDIR=/tmp/e208-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E208_on_policy_td_rehandle_stage_b_267k_v1/run.sh prepare

MPLCONFIGDIR=/tmp/e208-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E208_on_policy_td_rehandle_stage_b_267k_v1/run.sh \
  train-seed --through-episode 36 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e208-on-policy-td-rehandle-stage-b-267k-v1/run.log

MPLCONFIGDIR=/tmp/e208-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E208_on_policy_td_rehandle_stage_b_267k_v1/run.sh \
  evaluate --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e208-on-policy-td-rehandle-stage-b-267k-v1/eval.log

MPLCONFIGDIR=/tmp/e208-mpl \
bash experiments/conditioned_vcg/E208_on_policy_td_rehandle_stage_b_267k_v1/run.sh analyze
```
