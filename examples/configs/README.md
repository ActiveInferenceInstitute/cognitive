---
title: Example configurations
description: The settings file and generative model that example 01 loads.
type: documentation
status: stable
updated: 2026-10-13
semantic_relations:
  - type: documents
    links:
      - "[[../README|Examples]]"
      - "[[AGENTS|Example configuration instructions]]"
      - "[[../../docs/config/settings|Settings reference]]"
---

# Example configurations

- [`settings.yaml`](settings.yaml): `active_inference`, `simulation` and `output`
  values. Keys follow the [settings reference](../../docs/config/settings.md).
- [`two_state.yaml`](two_state.yaml): a generative model with `A`, `B_by_action`,
  `C`, `D`, `E` and `labels`. Load it with `cognitive.simulation.load_model` or
  `cognitive simulate --model`.

See [`../README.md`](../README.md) for the script that uses them.
