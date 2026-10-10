---
title: Development scripts
type: tool
status: stable
updated: 2026-10-10
created: 2026-08-02
tags:
  - scripts
  - development
  - tools
semantic_relations:
  - type: documents
    links:
      - "[[../../code/scripts/validate_docs.py|Scripts directory]]"
---

# Development scripts

The repository's executable scripts live in `code/scripts/`:

- `validate_docs.py` — documentation gate.
- `verify_links.py` — explicit wiki-link audit.
- `check_markdown_links.py` — standard Markdown link and anchor audit.
- `normalize_frontmatter.py` — `semantic_relations` link quoting.
- `kb_index.py` — knowledge-base catalog generation.
- `build_manuscript.py` — manuscript build pipeline.
- `mutation_check.py` — mutation gate.

Documentation-maintenance helpers live in `docs/repo_docs/repo_scripts/`
(`list_file_directory.py`, `fix_links.py`, `markdown_autofix.py`). Scripts
are also exposed as `cognitive-*` console commands where listed in
`pyproject.toml`.
