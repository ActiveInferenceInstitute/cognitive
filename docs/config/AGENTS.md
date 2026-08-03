---
title: Configuration documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-08-02
tags:
  - configuration
  - settings
  - parameters
  - agents
semantic_relations:
  - type: documents
    links:
      - '[[README|Configuration reference]]'
      - '[[simulation_config.yaml|Simulation example]]'
      - '[[../../config.yaml|Repository configuration]]'
---

# Configuration documentation instructions

The configuration documentation describes real configuration files and the
options accepted by the package's constructors and command-line interfaces.

## Ground truth

- The repository-level file is `config.yaml` at the repository root; its
  sections are documented in `docs/config/README.md`.
- `docs/config/simulation_config.yaml` is an example configuration, not a
  schema that every component loads.
- Runtime options come from Python constructors such as `InferenceConfig` and
  from the installed CLI entry points (`cognitive-* --help`).

## Rules

- Do not document generic configuration frameworks, feature-flag systems, or
  schema validators that are absent from this repository.
- When a configuration key is added or renamed, update
  `docs/config/README.md` in the same change.
- Verify claims against the loader or constructor that consumes the key.
