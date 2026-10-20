# E123: absolute flow-progress bound and matched scratch training

E123 preserves E122 and its failed progress-only results. It fixes the external-arrival reset loophole in a separately versioned controller and prepares a matched training comparison. This is a development experiment, not a paper-quality result or a replacement for an untouched test panel.

## What counts as progress

Only a **first admission into storage** or a **delivery** resets the absolute clock. Arrivals, Defer boundaries, and rehandles do not. The environment observes actual primitive execution, not a proposed action's label.

The critic and replay contain time since last progress, remaining absolute waiting allowance, the legacy Defer streak, and remaining committed-progress allowance, in addition to the existing public workload and guard inputs. All compared arms use the same extended input schema and identical initial parameters for a given model seed. No trained checkpoint initializes these runs.

## Quantitative liveness: conditions matter

Removing Defer alone is insufficient: Reconfigure could remain available forever. Also, recovery may require several macros before a delivery. E123 therefore uses:

- Absolute waiting allowance `W=30` primitive steps, never reset by external arrivals.
- A certified next-progress budget `B=20` steps, not an observed average macro length.
- Before expiry, nonprogress macros must fit the remaining waiting allowance and retain a bounded progress certificate.
- Direct Accept/Deliver macros must take at most 20 steps and retain bounded successor support.
- At expiry, only bounded direct progress is eligible. If none is available, the guard commits to a certified recovery prefix **through its first delivery**, including any necessary route clearing, with total duration at most 20 steps.
- An unsupported state fails closed. There is no fallback to uncertified actions or unrestricted waiting.

This is an additional quantitative liveness restriction, not merely a different Defer counter. It may exclude otherwise recoverable states. A failed search or unavailable bounded certificate is a failed gate, not proof that the underlying yard is infeasible.

For 36 empty-start jobs, there are 72 first-admission/delivery events. The conditional bound is `72*(30+20)=3600` steps. It requires first release at zero, release gaps at most 30, successful deterministic strict macro execution, and availability/preservation of the bounded progress support. When no work has been released, the controller cannot manufacture progress; that branch relies on the release-gap assumption. The preflight also checks bounded admission from the empty yard's start/exit landings. This is not a universal completion theorem for arbitrary arrivals or yard layouts. Runtime checks reject a gap exceeding 50 steps in the absolute-bound arm.

Both action selection and replay/Double-DQN backups use the same restricted frontier. All four cost heads bootstrap from one combined-cost-selected action, not four independently selected actions.

## Matched arms

| Arm | Feasibility | Waiting/liveness |
| --- | --- | --- |
| `A_outflow_bounded` | Original outflow certificate | E121 bounded inbound wait and original recovery guard |
| `B_joint_bounded` | E122 joint outflow/inflow certificate | Same as A |
| `C_joint_absolute` | Same joint certificate as B | New absolute progress shield above |

Start with A versus B to isolate inflow preservation. C is prepared but excluded from `train-matched`; it requires an explicit flag after reviewing A/B and the liveness gate. No certification-cost optimization is mixed into this comparison. All arms use exact goal-directed search, the same search budget, no relocation-family shortcut, and the same graph-input acceleration.

Training is from scratch, initially model seed 0, with seed 1 available for replication. Each arm uses the same arrival-driven empty 8x8/36-job instances: release gap 30, Poisson storage duration mean 80. Training seed origin is 187000000, with 10000-instance model-seed strides; the five reused development-validation seeds are 187800–187804. The integration stress uses 187990. These sets are disjoint. This experiment does not open the test panel.

The objective remains `J=2T+E+0.1L+10R`, with release-fixed due dates and a ±20 tolerance, gamma 1, and reward scale .01. Training epsilon decays .9→.05 over 8000 decisions; the learning rate is 5e-5 before decision 8000 and 1e-5 afterwards. Budget: 120 episodes per arm, validation every six episodes. Report greedy validation and the fixed final five checkpoints (96, 102, 108, 114, 120), not the most fortunate checkpoint. Full component metrics remain in the report. Twelve episodes are an integration checkpoint, not a learning-quality decision.

## Launch

Use the existing repository `.venv` and dependencies. CUDA is required for long training; deterministic CUDA configuration is set by `run.sh`. E119–E122's small authenticated source manifests and frozen diagnostic artifacts must remain available: they authenticate the inherited implementation and search setup, but their learned weights are not used for initialization. The large E118 replay checkpoint is not loaded.

```bash
cd /home/ai_diagnosis/HRL_Management
E123=experiments/conditioned_vcg/E123_absolute_progress_8x8_187k_v1/run.sh

# Clock/model/replay checks plus a deliberately waiting stress policy.
# No optimizer update is performed in that real-episode stress.
bash "$E123" preflight --device cpu

# Train both reference arms from scratch through the first checkpoint.
bash "$E123" train-matched --model-seed 0 --through-episode 12 --device cuda
bash "$E123" analyze

# Resume the exact saved states, exploration counters, replay and RNG.
bash "$E123" train-matched --model-seed 0 --through-episode 120 --device cuda
bash "$E123" analyze
```

If A/B justify testing the new liveness formulation:

```bash
bash "$E123" train-arm --arm C_joint_absolute --model-seed 0 \
  --through-episode 12 --device cuda --authorize-absolute-arm
```

For a second initialization, repeat the selected arm's command with `--model-seed 1`. The analysis command currently summarizes seed 0; seed 1 has its own complete histories in `seed-1/summary.json`. Do not describe one- or two-seed development comparisons as independent test evidence.

Every episode saves a resumable `latest.pth`; each six-episode endpoint also saves a small candidate checkpoint. Validation uses a frozen greedy clone and restores global RNG without advancing the training agent. A failed training episode is saved and halts the arm; it cannot be silently resumed past the failure. Code/contract changes require a new output version. The integration preflight includes an isolated synthetic optimizer/checkpoint test; this is not a research training run. The deliberately lazy stress policy tests the bound, not operational cost or learned quality.
