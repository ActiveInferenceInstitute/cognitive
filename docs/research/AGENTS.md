---
title: Research documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-08-02
tags:
  - research
  - agents
  - scientific
  - investigation
semantic_relations:
  - type: organizes
    links:
      - '[[README|Research documentation index]]'
      - '[[research_documentation|Research documentation guide]]'
      - '[[ant_colony_active_inference|Ant colony research]]'
      - '[[../repo_docs/getting_started|Getting started]]'
---

# Research documentation instructions

The research documentation covers methods, applications, and domain notes.
It is conceptual material; research claims must be grounded in the repository
or in cited external work.

## Ground truth

- `knowledge_base/research/` holds the research application notes and concept
  files that this folder links to.
- Executable validation lives in `code/tests/` and the manuscript's
  reproducibility section.

## Rules

- Do not invent results, statistics, citations, or experiment outputs.
- A research page that references a repository behavior must point at the real
  file or command that produces it.
- Pseudocode is allowed for illustration only when clearly labelled; it must
  not be presented as package API.
- Keep `research_documentation_index.md` and `index.md` consistent with the
  pages that exist.
