---
title: Configuration reference
type: documentation
status: stable
semantic_relations:
  - type: documents
    links:
      - '[[AGENTS|Configuration guidance]]'
      - '[[simulation_config.yaml|Simulation example]]'
      - '[[../../config.yaml|Repository configuration]]'
---

# Configuration reference

This directory contains configuration examples. It is not a separate runtime
configuration framework: the package's constructors and command-line interfaces
are the source of truth for supported options.

## Repository configuration

The root [`config.yaml`](../../config.yaml) records repository-level defaults and
paths used by the surrounding tooling. Its top-level sections are:

- `paths`: template, knowledge-base, model, utility, and data locations.
- `active_inference`: default precision, temporal horizon, learning rate,
  exploration factor, discount factor, inference iterations, and convergence
  threshold.
- `knowledge_base`: linking and relationship settings.
- `templates`: template defaults and required metadata fields.
- `visualization`: layout, seed, sizing, colors, and interactivity settings.
- `analysis`: metric names, logging level, and history retention.
- `obsidian`: vault synchronization and backup settings.
- `logging`: level, file, format, size, and backup count.
- `development`: debug, profiling, and test-mode switches.

These values are configuration data; not every key is consumed by every package
component. Check the relevant loader or constructor before relying on a key.

## Simulation example

[`simulation_config.yaml`](simulation_config.yaml) is an example configuration
for the repository's simulation-oriented documentation. Treat it as an example,
not as a guaranteed global schema. Validate any configuration against the
component that loads it.

## Runtime options

The public discrete runtime is configured with `InferenceConfig` in Python. The
root README demonstrates the supported fields used by the current dispatcher,
including `method`, `policy_type`, `temporal_horizon`, `learning_rate`,
`precision_init`, and `seed`. Model matrices are supplied to
`DiscreteGenerativeModel`; they are not read implicitly from `config.yaml`.

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
