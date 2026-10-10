---
title: Tools documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-10-10
tags:
  - tools
  - agents
  - utilities
  - automation
semantic_relations:
  - type: organizes
    links:
      - "[[docs/tools/README|Tools documentation index]]"
      - "[[docs/development/README|Development resources]]"
---

# Tools documentation instructions

The tools folder documents the tooling used by the repository: scripts,
validators, build steps, and automation utilities.

## Ground truth

- Executable tooling lives in `code/scripts/` and in the installed CLI entry
  points declared in `pyproject.toml`
  (`cognitive-create-node`, `cognitive-verify-links`, `cognitive-validate-docs`,
  `cognitive-normalize-frontmatter`, `cognitive-kb-index`, `cognitive-benchmark`,
  `cognitive-build-manuscript`, `cognitive-mutation-check`).
- The only CI workflow is `.github/workflows/quality.yml`. There is no
  Makefile, tox, pre-commit configuration, Sphinx or mkdocs build, or Docker
  image.
- Tool pages must describe real flags and outputs; run the tool before
  documenting it.

## Rules

- Do not document tools, frameworks, or automation engines that do not exist
  in the repository.
- Keep `README.md` in sync with the pages that exist; reduce pages for topics
  with no real tooling to a short statement of what the repository does use.
- When a script's interface changes, update its tool page in the same change.
