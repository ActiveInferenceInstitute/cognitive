---
title: Configuration documentation instructions
description: Rules for editing the configuration documentation so that it matches the loaders that read each key.
type: agents
status: stable
created: 2024-01-01
updated: 2026-10-13
tags:
  - configuration
  - settings
  - parameters
  - agents
semantic_relations:
  - type: documents
    links:
      - "[[README|Configuration reference]]"
      - "[[settings|Settings reference]]"
      - "[[simulation_config.yaml|Simulation example]]"
      - "[[../../config.yaml|Repository configuration]]"
---

# Configuration documentation instructions

The configuration documentation describes real configuration files and the
options accepted by the package's constructors and command-line interfaces.

## Ground truth

- The repository-level file is `config.yaml` at the repository root. Code reads
  these parts of it:
  - `active_inference`, `simulation` and `output`, through
    `cognitive.settings.load_settings` (`code/tools/src/settings.py`). The
    defaults, ranges and the `InferenceConfig` field each key sets are
    documented in [settings.md](settings.md), which a test checks against the
    code.
  - `paths.templates` and `paths.knowledge_base` (`NodeCreator`) and
    `visualization.seed` (`NetworkVisualizer`).
  - Every other section is descriptive, as [README.md](README.md) states.
- `docs/config/simulation_config.yaml` is an illustrative parameter sketch that
  no code loads.
- Runtime options come from Python constructors such as `InferenceConfig` and
  from the installed CLI entry points (`cognitive --help`, `cognitive-* --help`).
  The `cognitive` command is documented in
  [`../cli/reference.md`](../cli/reference.md).

## Rules

- Do not document generic configuration frameworks, feature-flag systems, or
  schema validators that are absent from this repository.
- When a configuration key is added, renamed, or begins to be consumed, update
  `docs/config/README.md` in the same change.
- Verify claims against the loader or constructor that consumes the key.
- Do not write that `active_inference.*` is descriptive or that nothing reads
  it: `cognitive.settings` reads it. The dispatcher itself takes an
  `InferenceConfig`, which `Settings.inference_config()` builds from the
  settings.
- A key added to `active_inference`, `simulation` or `output` needs a row in
  `settings.md` in the same change, with its type, its default and the
  `InferenceConfig` field it sets. Unknown keys in those sections are errors, so
  a documented key that is absent from the dataclasses in `settings.py` is wrong
  too.
- State defaults in backticks exactly as `Settings()` holds them (`1.0`, `3`,
  `variational`). `code/tests/test_docs_sync.py` searches the page for them.
- `settings.md` and `../cli/reference.md` quote real command output. After a
  change to a default or to what a command prints, rerun the command shown and
  replace the output.
