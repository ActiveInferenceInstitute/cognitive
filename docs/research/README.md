---
title: Research documentation index
type: documentation
status: stable
updated: 2026-10-10
semantic_relations:
  - type: organizes
    links:
      - "[[AGENTS|Research documentation instructions]]"
      - "[[index|Research index]]"
      - "[[research_documentation|Research documentation guide]]"
      - "[[ant_colony_active_inference|Ant colony research proposal]]"
      - "[[active_inference/README|Active inference research]]"
      - "[[architectures/README|Architectures research]]"
      - "[[complex_systems/README|Complex systems research]]"
      - "[[knowledge_base/research/README|Knowledge base research]]"
---

# Research documentation index

This directory holds orientation pages for the research themes of the project and
one design proposal. It is conceptual documentation: the theory is in
`knowledge_base/`, and the only executable research surface is the package
described in [[docs/api/api_reference|the API reference]].

## Pages

- [[index]] and [[research_documentation_index]]: navigation pages.
- [[research_documentation]]: where theory, code, and reproducibility checks live.
- [[ant_colony_active_inference]]: a design proposal for modelling foraging
  with the package (`status: proposal`; nothing in it has been run).

## Subfolders

- `active_inference/`: theory, applications, scaling, hierarchical extensions.
- `architectures/`: POMDP, continuous, hierarchical, multi-agent.
- `complex_systems/`: adaptation, collective behaviour, emergence,
  self-organization.
- `methodology/`: hypotheses, variables, sampling, validation as practised in
  this repository.
- `tools/`: the tooling used for analysis, documentation, and implementation.
- `examples/`, `guidelines/`, `implementations/`, `standards/`, `templates/`,
  `validation/`: short container pages that point to the real material.

Each leaf page states what the repository implements for its topic and links to
the knowledge-base notes that carry the theory.
