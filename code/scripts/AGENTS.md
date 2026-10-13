---
title: Repository script instructions
type: agents
status: stable
updated: 2026-10-13
semantic_relations:
  - type: documents
    links:
      - "[[README|Repository scripts]]"
      - "[[docs/development/README|Development]]"
---

# Repository scripts

Scripts are installed as the `scripts` package. Run them with `uv run python -m
scripts.<name>` from the repository root. Use real inputs and external-volume
output and temporary directories. Errors must retain their exit status.

`mutation_check` temporarily modifies source; run it alone after other tests finish.
`regenerate_gallery --publish` updates the committed page and ten images together.
Inspect generated figures before publishing and keep receipts under `output/`.
Test scripts with real temporary files and graphs in `code/tests/`.
