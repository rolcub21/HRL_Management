# Current repository layout

This is an index of the current tree, not a claim that the layout is ideal.

```text
HRL_Management/
├── README.md                       project overview and navigation
├── docs/                           documentation added before refactoring
│   └── UNIFIED_VCG.md              current unified-VCG evidence index
├── environment.py                 abstract environment
├── option.py                      abstract option
├── primitive_option.py            primitive-action adapter
├── options_agent.py               current manager/worker HRL agent
├── common_evaluate.py             common evaluation CLI
├── track_a_evaluate.py            strict assignment-isolation CLI
├── train_reg_selector.py          configurable flat REG-v3 trainer
├── train_reg_selector_v4.py       Track A-native set/candidate REG trainer
├── compare_pslap_baselines.py     matched-seed PSLAP geometry comparison
├── train_vcg_unified*.py          earlier unified-controller development trainers
├── evaluate_vcg_unified_*.py      earlier frozen-weight evaluation protocols
├── vcg_v11_nested_handling.py     current detached handling augmentation
├── run_vcg_v11_nested_*.py        current nested-VCG experiment protocols
├── plot_vcg_v11_nested_*.py       current confirmation/comparison figures
├── render_vcg_unified_behavior_gifs.py
│                                   authenticated qualitative replay renderer
├── experiments/                    versioned launchers and local protocol notes
├── run_learned_common_evaluation.sh
├── plot_revision_sensitivity.py   aggregate sensitivity plotting
├── plotting_test.py               legacy plot script
├── example/                        actual shipyard application + old examples
│   ├── small_rooms_env.py          shipyard environment
│   ├── episode_instance.py         immutable matched-evaluation input
│   ├── block_instance.py           block state model
│   ├── Options/                    operational and selector options
│   ├── helper/                     pathfinding, flattening, save utilities
│   ├── run_small_rooms_example_*.py
│   ├── genetic_algorithm.py        GA search implementation
│   ├── dqn.py, dqn_agent.py        earlier/alternative DQN code
│   ├── models/                     historical checkpoints
│   └── saved_models/               named saved weights
├── GA/                             GA yard and planning support
├── PSLAP/                          explicit dynamic and legacy PSLAP baselines
├── runs/hrl/                       TensorBoard artifacts
├── implementations/               generic option-generation algorithms
├── renderers/                      reusable environment renderers and assets
└── artifacts/legacy/evaluation/   small tracked historical smoke artifacts
```

## Source of truth by task

| Task | Start with |
|---|---|
| understand yard dynamics | `example/small_rooms_env.py` |
| understand block lifecycle | `example/block_instance.py` |
| understand hierarchical learning | `options_agent.py` |
| inspect operational options | `example/Options/` |
| train learned selector | `example/run_small_rooms_example_hrl.py` |
| train Track A-native REG-v4 | `train_reg_selector_v4.py`, then `PSLAP/reg_selector_v4.py` |
| train GA-assisted policy | `example/run_small_rooms_example_ga.py` |
| run PSLaP sensitivity | `example/run_small_rooms_example_pslap.py` |
| compare methods on a common seed | `common_evaluate.py` |
| isolate storage-assignment quality | `track_a_evaluate.py`, then `PSLAP/track_a.py` |
| inspect historical GA-assisted HRL internals | `example/genetic_algorithm.py`, then `GA/` |
| inspect independent PSLAP GA | `PSLAP/ga_optimizer.py`, then `PSLAP/ga_policy.py` |
| inspect repaired PSLAP internals | `PSLAP/online_policy.py`, `PSLAP/retrieval_dispatch.py`, then `PSLAP/dynamic_yard.py` |
| compare PSLAP assignment baselines | `compare_pslap_baselines.py` |
| inspect frozen PSLAP ablation | `PSLAP/legacy.py`, then `PSLAP/PSLAPPolicy.py` |
| follow the current unified VCG evidence | `docs/UNIFIED_VCG.md` |
| reproduce the current lambda-frontier confirmation | `experiments/vcg_v11_nested_lambda_confirmation_89k/` |
| reproduce the matched 89k comparison | `experiments/vcg_v11_nested_all_baselines_89k/` |
| render authenticated VCG behavior views | `render_vcg_unified_behavior_gifs.py` |

## Generated and historical material

The following should be treated as artifacts rather than source:

- `example/models/*.pth` and `example/saved_models/*.pth`;
- `runs/**/events.out.tfevents.*`;
- `artifacts/legacy/evaluation/` smoke CSVs and plot;
- root-level NPZ reward arrays;
- root-level PNG and EPS figures;
- all `__pycache__/` directories and `.pyc` files; and
- `render_log.txt`.

Large current results remain below ignored `results/` directories. The small
tracked legacy artifacts were grouped without changing their contents.

## Naming and status notes

- `example/` is misleading: it is the primary application.
- `GA/yard_enviromnet.py` contains a spelling error in its filename.
- `example/vizualization.py` contains a spelling error in its filename.
- `PSLAPPolicy.py` and several option filenames use inconsistent capitalization.
- `implementations/` contains generic option generators inherited from the
  earlier HRL code; `renderers/` contains reusable visualizers and taxi assets.
- Root-level research scripts remain in place because completed experiment
  contracts bind their paths and hashes. Future packaging should therefore be
  versioned rather than performed as a silent move.
