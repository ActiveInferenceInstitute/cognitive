---
title: Cognitive Agent Architectures
type: agents
status: active
created: 2024-01-01
updated: 2026-08-02
tags:
  - cognitive
  - agents
  - architectures
  - active_inference
  - implementation
semantic_relations:
  - type: organizes
    links:
      - '[[README|Cognitive science knowledge base]]'
      - '[[active_inference|Active inference]]'
      - '[[active_inference_agent|Active inference agent]]'
      - '[[cognitive_architecture|Cognitive architecture]]'
      - '[[free_energy_principle|Free energy principle]]'
      - '[[predictive_processing|Predictive processing]]'
      - '[[belief_updating|Belief updating]]'
---

# Cognitive Agent Architectures

The cognitive domain is the largest knowledge-base section, covering
cognitive science, neuroscience, and psychology through the lens of active
inference. Its `README.md` is the navigation entry point; this page records
the agent-architecture focus of the domain.

## Documents

The folder contains the domain's concept pages (approximately 170 files; see
`README.md` for the full navigation). Core entries include:

- [[active_inference]] and [[active_inference_agent]] — the theory and agent
  formulations.
- [[cognitive_architecture]] — architecture patterns.
- [[free_energy_principle]] and [[predictive_processing]] — foundations.
- [[belief_updating]] — inference mechanisms.

## Notes

- The pages are conceptual. Pseudocode within them is illustrative and is not
  package API; executable behavior lives in the `cognitive` and `Things`
  packages.
- Authoring rules for knowledge-base pages are in
  `knowledge_base/AGENTS.md`.
