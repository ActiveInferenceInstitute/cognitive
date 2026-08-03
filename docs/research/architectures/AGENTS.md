---
title: Architectures research instructions
type: agents
status: stable
semantic_relations:
  - type: organizes
    links:
      - '[[README|Architectures research]]'
      - '[[../README|Research documentation]]'
      - '[[continuous|Continuous architectures]]'
      - '[[hierarchical|Hierarchical architectures]]'
      - '[[multi_agent|Multi-agent architectures]]'
      - '[[pomdp|POMDP architectures]]'
---

# Architectures research instructions

This folder holds research notes on agent architectures: continuous,
hierarchical, multi-agent, and POMDP treatments.

## Rules

- Notes are conceptual; the executable agents live in `code/Things/` and in
  `code/tools/src/models/active_inference/`.
- Do not present illustrative classes as available interfaces.
- Keep `README.md` in sync with the pages that exist.
- Run `python code/scripts/validate_docs.py --json` before merging.
