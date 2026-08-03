---
title: Manuscript instructions
type: agents
status: stable
semantic_relations:
  - type: organizes
    links:
      - '[[README|Manuscript overview]]'
      - '[[SYNTAX|Manuscript syntax]]'
      - '[[config.yaml|Manuscript configuration]]'
      - '[[../api/README|API documentation]]'
---

# Manuscript instructions

This directory is the publication source for the repository's executable
manuscript. It is build input, not loose prose: the numbered sections,
labels, citations, and figures are validated by
`code/scripts/validate_docs.py` and hydrated by
`code/scripts/build_manuscript.py`.

## Rules

- Keep the numbered-section convention: lexicographic order is publication
  order (`00_abstract.md` through `99_references.md`).
- Do not add figures that the builder does not generate; figure references
  are checked against `build_manuscript.py`.
- Keep labels (`{#eq:...}`, `{#fig:...}`, `{#tbl:...}`, `{#sec:...}`) and
  citation keys unique and resolvable; keep `references.bib` in sync.
- `SYNTAX.md` records the tokens and contract; `layer_contract.yaml` records
  which executable files may back which claims.
- After editing, run the build and the documentation gate:
  `cognitive-build-manuscript --output build/manuscript` and
  `python code/scripts/validate_docs.py --json`.
