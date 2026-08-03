---
title: Configuration tools
type: tool
status: stable
created: 2026-08-02
tags:
  - configuration
  - tools
  - yaml
semantic_relations:
  - type: documents
    links:
      - '[[../config/README|Configuration reference]]'
---

# Configuration tools

Configuration in this repository is handled by the package's constructors and
CLI entry points, not by a generic config framework:

- Repository-level defaults: `config.yaml` (sections documented in
  `docs/config/README.md`).
- Runtime model configuration: `InferenceConfig` in Python (see the root
  `README.md` example).
- CLI options: `cognitive-* --help` for each installed entry point.

There is no `ConfigManager` class or schema-validator API; do not document
one. When a configuration key is added or renamed, update
`docs/config/README.md` in the same change.
