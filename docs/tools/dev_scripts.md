---
title: Development scripts
type: tool
status: stable
created: 2026-08-02
tags:
  - scripts
  - development
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../../code/scripts/validate_docs.py|Scripts directory]]'
---

# Development scripts

The repository's executable scripts live in `code/scripts/`:

- `validate_docs.py` — documentation gate.
- `verify_links.py` — explicit wiki-link audit.
- `check_markdown_links.py` — standard Markdown link and anchor audit.
- `build_manuscript.py` — manuscript build pipeline.

Documentation-maintenance helpers live in `docs/repo_docs/repo_scripts/`
(`list_file_directory.py`, `fix_links.py`, `markdown_autofix.py`). Scripts
are also exposed as `cognitive-*` console commands where listed in
`pyproject.toml`.
