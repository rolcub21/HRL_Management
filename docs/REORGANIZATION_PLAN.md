# Proposed non-breaking-to-breaking reorganization

## Implemented non-breaking compatibility layer

The current final method now has a stable public interface at
`methods/conditioned_vcg/`. Paper-facing experiments begin with E1 at
`experiments/conditioned_vcg/`; the architecture-selection history is indexed
separately as D1--D9. The robustness extension has its own
`experiments/robust_vcg/` index.

The original root modules and flat experiment directories remain installed as
frozen evidence sources. Numbered launchers delegate to them rather than
moving them, because completed contracts authenticate their exact paths and
byte hashes. This implements the package-and-compatibility-wrapper step below
without invalidating prior results.

## Scope of the remaining migration

The current phase intentionally does not move or edit authenticated executable
Python files. It establishes terminology, identifies sources of truth, and
records experiment inputs and outputs while making structural risks explicit.

Moving source now would require import changes. Moving historical artifacts may
also invalidate paths in scripts, notebooks, or unpublished experiment notes.

## Proposed target layout

```text
shipyard-hrl/
├── README.md
├── pyproject.toml
├── configs/
│   ├── hrl_sensitivity.yaml
│   ├── ga_sensitivity.yaml
│   └── pslap_sensitivity.yaml
├── docs/
├── src/shipyard_hrl/
│   ├── core/
│   │   ├── environment.py
│   │   ├── option.py
│   │   └── agent.py
│   ├── envs/
│   │   ├── shipyard.py
│   │   └── block.py
│   ├── options/
│   ├── methods/
│   │   ├── hrl/
│   │   ├── ga/
│   │   └── pslap/
│   ├── evaluation.py
│   └── plotting.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── plot_results.py
├── tests/
├── results/                ignored; generated experiment directories
└── artifacts/              external or curated inputs with manifests
```

## Migration map

| Current location | Proposed destination |
|---|---|
| `environment.py` | `src/shipyard_hrl/core/environment.py` |
| `option.py` | `src/shipyard_hrl/core/option.py` |
| `primitive_option.py` | `src/shipyard_hrl/core/primitive_option.py` |
| `options_agent.py` | `src/shipyard_hrl/core/agent.py` |
| `example/small_rooms_env.py` | `src/shipyard_hrl/envs/shipyard.py` |
| `example/block_instance.py` | `src/shipyard_hrl/envs/block.py` |
| `example/Options/` | `src/shipyard_hrl/options/` |
| `GA/` and active GA code | `src/shipyard_hrl/methods/ga/` |
| `PSLAP/` | `src/shipyard_hrl/methods/pslap/` |
| three run scripts | one configuration-driven `scripts/train.py` |
| `common_evaluate.py` | library evaluator plus `scripts/evaluate.py` |
| root generated files | timestamped directories under `results/legacy/` |

## Safe implementation order

1. Add characterization tests against current environment resets, seeded
   episodes, state dimensions, option sets, and checkpoint loading.
2. Capture a small set of reference outputs before changing imports.
3. Create an installable package while retaining temporary compatibility
   wrappers at old entry points.
4. Move one subsystem at a time and run characterization tests after each move.
5. Replace embedded experiment settings with validated configuration files and
   CLI overrides.
6. Introduce experiment manifests recording code revision, seeds, parameters,
   dependency versions, artifact hashes, and output schema version.
7. Move legacy outputs only after generating a machine-readable mapping from old
   paths to new paths.
8. Remove compatibility wrappers only after all documented commands use the new
   package.

## Decisions to make before code refactoring

- Whether the official project name is `shipyard-hrl`, `hrl-eval`, or
  `HRL_Management`.
- Whether GA-assisted HRL is called `GA`, `GA-HRL`, or another unambiguous name.
- Delivery error is a signed timing offset. Accuracy, earliness, and tardiness
  are reported separately by the common evaluator.
- Whether the two timing features described in `get_current_state()` should be
  restored or removed from its docstring permanently.
- Which DQN implementation is authoritative and which files are archival.
- Which historical checkpoints must remain loadable.
- Whether the full PSLaP grid or only diagonal settings constitute the primary
  comparison.

No existing file should be deleted solely because it appears unused until those
decisions and the characterization tests are in place.
