# VCG 1.1-anchored preference pilot (seed 0)

This is the conservative replacement experiment for the unsuccessful
from-scratch preference-conditioned architecture B.

The controller is

\[
Q_{op}(s,c,\lambda)=Q_{op}^{V1.1}(s,c)
  + \frac{\lambda}{\lambda_{max}}\Delta Q_{op}(s,c,\lambda),
\qquad
M_\lambda=Q_{op}-\lambda Q_N.
\]

The authenticated VCG 1.1 encoder and operational critic are frozen. At
`lambda=0`, selection delegates directly to VCG 1.1, so this endpoint cannot
be degraded by residual training. The residual operational tower starts at
exactly zero. The conditioned handling tower starts as an exact copy of the
completed detached Monte-Carlo handling head, so every positive-lambda policy
also starts at the existing nested VCG operating point.

Only the two small conditioned towers train. Operational return keeps VCG
1.1's primitive-time discount; handling predicts the undiscounted number of
physical rehandles, which is the reported resource metric. Frozen VCG
features are cached in replay.

## Single-seed sequence

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/vcg_v11_anchored_preference_seed0_85k/run.sh run-seed0
```

The command is sequential and resumable. It performs:

1. contract preparation;
2. five short sentinel rollouts proving the initial policy matches VCG 1.1 at
   `lambda=0` and the existing nested policy at positive lambdas;
3. 200 seed-0 training episodes on fresh 53,000,000-series instances;
4. one post-training lambda-zero sentinel plus 48 positive-lambda rollouts on
   the already-opened 85k development panel;
5. a behavioral and trade-off report.

To resume an interrupted run, invoke the same `run-seed0` command. The script
detects `latest.pth` and resumes at the next clean episode boundary.

The final diagnostic is written to:

`results/vcg-v1-1-anchored-preference-seed0-85k-development/anchored-seed0-report.json`

This is a one-seed, opened-panel development result. Seeds 1 and 2 are
intentionally not launched until seed 0's behavior is inspected.
