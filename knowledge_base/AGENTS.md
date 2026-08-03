---
title: Knowledge Base Active Inference Documentation
type: agents
status: stable
created: 2024-01-01
updated: 2026-08-02
tags:
  - active_inference
  - knowledge_base
  - documentation
  - cognitive_modeling
semantic_relations:
  - type: organizes
    links:
      - '[[README|Knowledge base overview]]'
      - '[[index|Knowledge base index]]'
      - '[[cognitive/README|Cognitive science]]'
      - '[[mathematics/README|Mathematics]]'
      - '[[biology/README|Biology]]'
      - '[[systems/README|Systems theory]]'
      - '[[philosophy/README|Philosophy]]'
      - '[[ontology/README|Ontology]]'
      - '[[agents/README|Agent architectures]]'
      - '[[free_energy_principle/README|Free energy principle]]'
---

# Knowledge Base Active Inference Documentation

The knowledge base is the repository's conceptual layer: it collects
theoretical foundations, domain notes, and research applications related to
Active Inference and cognitive modeling. It is an Obsidian vault of
interconnected Markdown pages, not an executable package.

## Relationship to the package

- Executable behavior is defined by the installed `cognitive` and `Things`
  packages and their tests; the knowledge base explains concepts.
- Concept pages may illustrate ideas with pseudocode, but those snippets are
  not package API.
- Root reference pages: `glossary.md`, `learning_roadmap.md`,
  `linking_standards.md`, `quality_assessment.md`,
  `swarm_intelligence_implementation.md`.

## Authoring rules

- Pages use YAML frontmatter and `[[wiki links]]` for cross-references;
  explicit file-like links must resolve (checked by
  `code/scripts/verify_links.py`).
- Do not invent citations, statistics, or references; cite the actual
  literature or the actual repository files.
- Domain overviews live in each domain's `README.md`; each content folder
  carries a short `AGENTS.md` index in the style of the folder files.
