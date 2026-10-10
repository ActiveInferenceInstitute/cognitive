---
title: Research documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-10-10
tags:
  - research
  - agents
semantic_relations:
  - type: organizes
    links:
      - "[[README|Research documentation index]]"
      - "[[research_documentation|Research documentation guide]]"
      - "[[ant_colony_active_inference|Ant colony research proposal]]"
      - "[[../repo_docs/getting_started|Getting started]]"
---

# Research documentation instructions

The research documentation holds orientation pages and one design proposal. The
theory itself lives in `knowledge_base/`; the executable surface is listed in
[[docs/api/api_reference|the API reference]].

## Ground truth

- `knowledge_base/research/` holds the research application notes and concept
  files that this folder links to.
- Executable checks live in `code/tests/`, in `cognitive-benchmark`, and in the
  manuscript build (`cognitive-build-manuscript`, see
  [[docs/manuscript/06_reproducibility|the reproducibility chapter]]).
- The package has no research-specific API and no ant colony, hierarchical, or
  multi-agent implementation. Pages here say so rather than implying otherwise.

## Rules

- Do not invent results, statistics, citations, or experiment outputs.
- A page that references repository behaviour must point at the real file or
  command that produces it.
- Illustrative code must either run against the package or be labelled as a
  design sketch; it must not be presented as package API.
- A page that describes work that has not been built carries
  `status: proposal` in its frontmatter.
- Keep `research_documentation_index.md` and `index.md` consistent with the
  pages that exist.
