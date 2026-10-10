---
title: Biological Knowledge Base Documentation
type: agents
status: stable
created: 2025-01-01
updated: 2026-08-02
tags:
  - agents
  - biology
  - knowledge_base
  - ecological
  - evolutionary
  - active_inference
semantic_relations:
  - type: organizes
    links:
      - "[[README|Biology knowledge base overview]]"
      - "[[behavioral_biology|Behavioral biology]]"
      - "[[evolutionary_biology|Evolutionary biology]]"
      - "[[homeostasis|Homeostasis]]"
      - "[[ecological_networks|Ecological networks]]"
      - "[[ant_colony_organization|Ant colony organization]]"
---

# Biological Knowledge Base Documentation

The biology domain collects conceptual documents on biological principles and
systems relevant to active inference and cognitive modeling: evolution,
homeostasis, ecological networks, collective behavior, and neural systems.

## Documents

The folder contains the domain's concept pages (see `README.md` for the full
navigation). Key entries include behavioral and evolutionary biology,
homeostasis, physiology, ecological networks, and social-insect organization.

## Notes

- This material is conceptual; it explains biology through the free energy
  principle. Pseudocode within pages is illustrative.
- No biological model is importable: the `Ant_Colony` and `BioFirm` folders
  in `code/Things/` hold design notes only. The runnable agents are the
  `cognitive` package, `Simple_POMDP`, and `Continuous_Generic`.
- Authoring rules for knowledge-base pages are in
  `knowledge_base/AGENTS.md`.
