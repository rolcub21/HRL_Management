E147 WAIT-AWARE WORKING-DAY TRAINING
====================================

Primary objective:
  J = 2*T + E + 0.5*W + 0.1*L + 10*R

W is charged densely for every released block until its first completed
storage admission. T remains dense after due+20; E is charged on delivery
before due-20. Completion remains a hard eligibility condition.

The source is the successful E131/E133 residual seed-0 episode-60 model. Its
graph encoder, context encoder, consequence trunk and four cost heads are
copied exactly. A fifth admission-wait residual head and a three-input arrival
projection start at exactly zero. Optimizer, replay and adaptation counters are
fresh. All heads bootstrap from one combined-J selected successor.

Policy-visible arrival context contains only:
  - releases observed in the trailing 120 steps divided by 120;
  - steps since the last observed release (or episode start);
  - current working-day phase, clipped at one.
The sampled overlap, total jobs, simulator arrival rate, regime identity and
future release schedule are not inputs.

Every training episode starts with the original empty 8x8 yard. Day length is
5400, storage durations are positive Poisson(1200), and nominal expected
overlap is sampled continuously from Uniform[10%,80%]. Arrivals occur during
the day; all jobs must be delivered afterward. Validation uses fixed
development days at 20%, 50% and 80%. The independent test panel remains
sealed.

The E146 empty-frontier failure is repaired narrowly: after the Defer streak
cap, another bounded Defer remains legal only if the episode is nonterminal
and no released work exists. The controller observes no future release time.
Ordinary admission, due, guard and Defer restrictions resume immediately when
work is observable.

Short local commands (Codex runs these):
  MPLCONFIGDIR=/tmp/e147-mpl .venv/bin/python -B -m \
    experiments.conditioned_vcg.E147_wait_aware_working_day_206k_v1.run prepare
  MPLCONFIGDIR=/tmp/e147-mpl .venv/bin/python -B -m \
    experiments.conditioned_vcg.E147_wait_aware_working_day_206k_v1.run smoke

LONG USER-LAUNCHED RUN, first checkpoint:
  cd /home/ai_diagnosis/HRL_Management
  mkdir -p results/vcg-conditioned-e147-wait-aware-working-day-206k-v1-repair3
  MPLCONFIGDIR=/tmp/e147-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    .venv/bin/python -u -B -m \
    experiments.conditioned_vcg.E147_wait_aware_working_day_206k_v1.run \
    train --through-episode 12 --device cuda 2>&1 | tee -a \
    results/vcg-conditioned-e147-wait-aware-working-day-206k-v1-repair3/run.log

After inspecting episode 12, resume the SAME command with 60. Episode 60 is
the predeclared decision point. Episode 120 is available only as a later,
explicit continuation; it is never started automatically. Checkpoints are
resumable and endpoint evaluation snapshots are retained.

Monitor without stopping training:
  tail -f /home/ai_diagnosis/HRL_Management/results/\
vcg-conditioned-e147-wait-aware-working-day-206k-v1-repair3/run.log

Ctrl-C stops tail only. To stop training, interrupt its own training terminal.
SIGTERM and Ctrl-C retain the last completed-episode checkpoint and record the
unfinished episode as discarded in stopped.json.
