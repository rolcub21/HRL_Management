# Current repository layout

This is an index of the current tree, not a claim that the layout is ideal.

```text
hrl-eval/
├── README.md                       project overview and navigation
├── docs/                           documentation added before refactoring
├── environment.py                 abstract environment
├── option.py                      abstract option
├── primitive_option.py            primitive-action adapter
├── options_agent.py               current manager/worker HRL agent
├── common_evaluate.py             common evaluation CLI
├── run_learned_common_evaluation.sh
├── plot_revision_sensitivity.py   aggregate sensitivity plotting
├── plotting_test.py               legacy plot script
├── example/                        actual shipyard application + old examples
│   ├── small_rooms_env.py          shipyard environment
│   ├── block_instance.py           block state model
│   ├── Options/                    operational and selector options
│   ├── helper/                     pathfinding, flattening, save utilities
│   ├── run_small_rooms_example_*.py
│   ├── genetic_algorithm.py        GA search implementation
│   ├── dqn.py, dqn_agent.py        earlier/alternative DQN code
│   ├── models/                     historical checkpoints
│   └── saved_models/               named saved weights
├── GA/                             GA yard and planning support
├── PSLAP/                          PSLaP heuristic implementation
├── runs/hrl/                       TensorBoard artifacts
├── implementations/               currently empty
├── renderers/                      currently empty
└── root-level *.csv/*.npz/images   historical generated artifacts
```

## Source of truth by task

| Task | Start with |
|---|---|
| understand yard dynamics | `example/small_rooms_env.py` |
| understand block lifecycle | `example/block_instance.py` |
| understand hierarchical learning | `options_agent.py` |
| inspect operational options | `example/Options/` |
| train learned selector | `example/run_small_rooms_example_hrl.py` |
| train GA-assisted policy | `example/run_small_rooms_example_ga.py` |
| run PSLaP sensitivity | `example/run_small_rooms_example_pslap.py` |
| compare methods on a common seed | `common_evaluate.py` |
| inspect GA internals | `example/genetic_algorithm.py`, then `GA/` |
| inspect PSLaP internals | `PSLAP/PSLAPPolicy.py`, then `PSLAP/yard_logic.py` |

## Generated and historical material

The following should be treated as artifacts rather than source:

- `example/models/*.pth` and `example/saved_models/*.pth`;
- `runs/**/events.out.tfevents.*`;
- root-level evaluation CSVs and debug logs;
- root-level NPZ reward arrays;
- root-level PNG and EPS figures;
- all `__pycache__/` directories and `.pyc` files; and
- `render_log.txt`.

They have deliberately not been moved yet because some scripts and external
records may refer to their present locations.

## Naming and status notes

- `example/` is misleading: it is the primary application.
- `GA/yard_enviromnet.py` contains a spelling error in its filename.
- `example/vizualization.py` contains a spelling error in its filename.
- `PSLAPPolicy.py` and several option filenames use inconsistent capitalization.
- `implementations/` and `renderers/` are empty and have no documented runtime
  role.
- The project root has no standalone `.git` directory.
