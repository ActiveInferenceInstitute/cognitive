---
title: Tools documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-08-02
tags:
  - tools
  - agents
  - utilities
  - automation
semantic_relations:
  - type: organizes
    links:
      - '[[README|Tools documentation index]]'
      - '[[development_tools_index|Development tools index]]'
      - '[[../../code/scripts/validate_docs.py|Repository scripts]]'
---

# Tools documentation instructions

The tools folder documents the tooling used by the repository: scripts,
validators, build steps, and automation utilities.

## Ground truth

- Executable tooling lives in `code/scripts/` and in the installed CLI entry
  points declared in `pyproject.toml`
  (`cognitive-create-node`, `cognitive-verify-links`, `cognitive-validate-docs`,
  `cognitive-benchmark`, `cognitive-build-manuscript`).
- Tool pages must describe real flags and outputs; run the tool before
  documenting it.

## Rules

- Do not document tools, frameworks, or automation engines that do not exist
  in the repository.
- Keep `README.md` and `development_tools_index.md` in sync with the pages
  that exist; merge or remove pages for topics with no real tooling.
- When a script's interface changes, update its tool page in the same change.
