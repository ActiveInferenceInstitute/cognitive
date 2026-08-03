---
title: Cursor integration
type: tool
status: stable
created: 2026-08-02
tags:
  - cursor
  - ide
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../../.cursorrules|Cursor rules]]'
      - '[[../AGENTS|Tools documentation instructions]]'
---

# Cursor integration

This repository is set up for Cursor (and similar AI-assisted editors)
through:

- `.cursorrules` at the repository root — the project's Cursor rules:
  documentation requirements (AGENTS.md/README.md per folder, Obsidian
  linking, `semantic_relations` frontmatter), code quality standards
  (test-driven development, no fixture-only tests), and conventions.
- `AGENTS.md` and `CLAUDE.md` at the root — the same guidance in
  agent-portable form.
- YAML frontmatter and `[[wiki links]]` throughout the vault, which
  AI-assisted editors use for navigation.

There is no repository-specific Cursor plugin; the integration is the
rules file plus the documented development loop in
`docs/development/README.md`. Keep `.cursorrules` in sync with
`AGENTS.md`/`CLAUDE.md` when conventions change.
