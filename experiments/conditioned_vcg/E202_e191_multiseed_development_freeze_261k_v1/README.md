# E202 — E191 multiseed through-120 development freeze

E202 authenticates E191 model seeds 0, 1, and 2 after each has completed the
predeclared through-120 development run. It freezes each trainer-selected
candidate byte-for-byte. Selection remains E191's predeclared lexicographic
development score; E202 does not substitute the terminal checkpoint or impose
the seed-0-specific E194 assumption that the winner is episode 100 or 120.

The selected episodes are 100, 60, and 80 for model seeds 0, 1, and 2. These
are development selections from a paused 300-episode protocol. They are not
deployment-eligible, do not open a test panel, and are not paper evidence.

```bash
bash experiments/conditioned_vcg/E202_e191_multiseed_development_freeze_261k_v1/run.sh \
  freeze-all
```

Outputs are ignored by Git and written beneath:

```text
results/vcg-conditioned-e202-e191-multiseed-development-freeze-261k-v1/
  freeze.json
  seed-0/{freeze.json,selected-development-checkpoint.pth}
  seed-1/{freeze.json,selected-development-checkpoint.pth}
  seed-2/{freeze.json,selected-development-checkpoint.pth}
```
