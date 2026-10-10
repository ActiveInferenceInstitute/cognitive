---
title: Generic Thing Design Notes
type: design_notes
status: draft
created: 2025-01-01
updated: 2026-10-10
tags:
  - generic_thing
  - markov_blanket
  - message_passing
  - design
semantic_relations:
  - type: relates
    links:
      - "[[code/Things/Generic_Thing/README|Generic Thing folder]]"
      - "[[knowledge_base/cognitive/free_energy_principle|Free energy principle]]"
      - "[[knowledge_base/cognitive/hierarchical_models|Hierarchical models]]"
      - "[[knowledge_base/cognitive/multi_agent_active_inference|Multi-agent active inference]]"
---

# Generic Thing: Design Notes

> **Status: design sketch; no code in this repository.** `code/Things/Generic_Thing/` contains `README.md`, `AGENTS.md`, this page and a `generic_thing/visualizations/` directory. That directory holds only PNG images (for example `markov_blanket.png`, `free_energy_landscape.png`, `message_network_*.png`, sensor/actuator/processor repertoire plots, with timestamps from 2025-02-23). It contains no Python. There is no `GenericThing`, `MarkovBlanket`, `FreeEnergy`, `MessagePassing` or `FederatedInference` class, no `core.py`, `requirements.txt` or test suite, and no benchmarks. The images are retained outputs of an earlier experiment whose code is not present, so they cannot be regenerated from this repository.

## Concept

A "Thing" is the most abstract Active Inference entity: something with a boundary (a Markov blanket) that separates internal states from external states through sensory and active states. Things can contain other Things, which gives nested blankets and a hierarchy in which each part is itself a whole.

## Ideas the design draws on

- **Free energy principle:** a Thing acts to minimize variational free energy, which bounds surprise.
- **Nested Markov blankets:** conditional independence at each level of the hierarchy.
- **Message passing:** beliefs propagate between connected Things; Things exchange observations, actions, beliefs and parameters.
- **Federated inference:** several Things reach consensus by sharing evidence or parameters.

## Proposed update cycle

1. Receive observations through the blanket's sensory states.
2. Update the blanket and the internal state.
3. Minimize free energy to update beliefs.
4. Select actions through the blanket's active states.
5. Propagate messages to connected Things.

## What is available instead

- `code/Things/Simple_POMDP/`: discrete-state Active Inference agent with tests.
- `code/Things/Continuous_Generic/`: continuous-state agent.
- [[docs/api/api_reference|API reference]] and [[docs/agents/agent_docs_readme|agent documentation]] for the exported API.
- Theory: [[knowledge_base/cognitive/free_energy_principle|free energy principle]], [[knowledge_base/cognitive/hierarchical_models|hierarchical models]], [[knowledge_base/cognitive/multi_agent_active_inference|multi-agent active inference]].
