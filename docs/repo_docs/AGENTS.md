---
title: Repository documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-08-02
tags:
  - repository
  - documentation
  - standards
  - maintenance
semantic_relations:
  - type: organizes
    links:
      - '[[README|Repository documentation index]]'
      - '[[documentation_standards|Documentation standards]]'
      - '[[ai_documentation_style|Documentation style]]'
      - '[[content_management|Content management]]'
      - '[[linking_validation|Link validation]]'
      - '[[getting_started|Getting started]]'
---

# Repository documentation instructions

The `repo_docs/` folder records repository-wide documentation standards,
maintenance practices, linking rules, and the getting-started guide.

## Ground truth

- The standards pages (`documentation_standards.md`,
  `ai_documentation_style.md`, `naming_conventions.md`,
  `obsidian_linking.md`) describe conventions; the enforceable checks are the
  repository's scripts in `code/scripts/` and `repo_scripts/`.
- `getting_started.md` is the fresh-checkout guide and must match the root
  `README.md` and `pyproject.toml`.

## Rules

- Standards must match the tools that enforce them
  (`code/scripts/validate_docs.py`, `code/scripts/verify_links.py`,
  `.markdownlint.json`).
- The helper scripts in `repo_scripts/` are utilities; document their real
  flags and outputs.
- Do not describe documentation automation frameworks that do not exist here.
