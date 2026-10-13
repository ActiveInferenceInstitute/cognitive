---
title: Configuration reference
type: documentation
status: stable
updated: 2026-10-09
semantic_relations:
  - type: documents
    links:
      - "[[AGENTS|Configuration guidance]]"
      - "[[simulation_config.yaml|Simulation example]]"
      - "[[../../config.yaml|Repository configuration]]"
---

# Configuration reference

This directory documents the repository's configuration files. It is not a
separate runtime configuration framework: the package's constructors and
command-line interfaces are the source of truth for supported options.

## Repository configuration

The root [`config.yaml`](../../config.yaml) has these top-level sections:

- `paths`: template, knowledge-base, model, utility, and data locations.
- `active_inference`: default precision, temporal horizon, learning rate,
  exploration factor, and discount factor.
- `simulation`: inference method, policy aggregation, temperature, policy limit,
  sample count, episode length, and seed.
- `output`: result directory, figure format, and figure resolution.
- `knowledge_base`: linking and relationship settings.
- `templates`: template defaults and required metadata fields.
- `visualization`: layout, seed, sizing, colors, and interactivity settings.
- `analysis`: metric names, logging level, and history retention.
- `obsidian`: vault synchronization and backup settings.
- `logging`: level, file, format, size, and backup count.
- `development`: debug, profiling, and test-mode switches.

Only these keys are read by code in this repository:

- `active_inference.*`, `simulation.*` and `output.*`, by
  `cognitive.settings.load_settings`, which the `cognitive` command and the
  [`examples/`](../../examples/README.md) use. [`settings.md`](settings.md)
  documents every key, its default, the layering order and the validation rules.
- `paths.templates` and `paths.knowledge_base`, by `NodeCreator`
  (`cognitive-create-node`); relative paths resolve against the directory that
  holds the YAML file.
- `paths.knowledge_base` and `visualization.seed`, by `NetworkVisualizer`.

Every other key in `config.yaml` (including `analysis`, `obsidian`, `logging`,
and `development`) is descriptive: no loader consumes it. The dispatcher itself
does not read `config.yaml`; it takes an `InferenceConfig`, which
`Settings.inference_config()` builds from the settings.

## Simulation example

[`simulation_config.yaml`](simulation_config.yaml) holds illustrative BioFirm
simulation parameters (time steps, hierarchies, intervention strategies). No
code in the repository loads it, and `code/Things/BioFirm/` contains
documentation only. Treat it as a sketch of a parameter set, not as a schema.

## Runtime options

The public discrete runtime is configured with `InferenceConfig` in Python. The
required fields are `method`, `policy_type`, `temporal_horizon`, `learning_rate`,
and `precision_init`; the optional fields are listed in
[`../api/api_reference.md`](../api/api_reference.md). Model matrices are
supplied to `DiscreteGenerativeModel`; they are not read implicitly from
`config.yaml`.

Other loaders take their own schemas, and reject unknown keys:

- `ActiveInferenceFactory.create_from_yaml`: dispatcher fields plus a
  `generative_model` mapping with `A`, `B`, `C`, `D`, `E`.
- `SimplePOMDP`: sections `model`, `state_space`, `observation_space`,
  `action_space`, `matrices`, `inference`, `visualization`, and optional `seed`.
- `HomeostaticFactory`: sections `state_spaces`, `observation_model`,
  `transition_model`, `inference`, and optional `target_state`,
  `initial_beliefs`, `action_prior`, `seed`.
- The manuscript builder reads `docs/manuscript/config.yaml`.

Command-line options are discoverable from the installed entry points:

```bash
cognitive-create-node --help
cognitive-benchmark --help
cognitive-build-manuscript --help
cognitive-validate-docs --help
cognitive-verify-links --help
```

## Adding configuration

When adding a configuration key:

1. Identify the loader or constructor that consumes it.
2. Document its type, default, and accepted values next to the example.
3. Add or update a test that exercises the behavior.
4. Run the Python and documentation gates from
   [`docs/development/README.md`](../development/README.md).

Do not document generic `ConfigManager`, `ConfigValidator`, feature-flag, or
schema APIs unless those symbols are implemented and publicly importable.
