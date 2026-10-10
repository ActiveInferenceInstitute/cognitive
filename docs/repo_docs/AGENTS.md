---
title: Repository documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-10-10
tags:
  - repository
  - documentation
  - standards
  - maintenance
semantic_relations:
  - type: organizes
    links:
      - "[[docs/repo_docs/README|Repository documentation index]]"
      - "[[docs/repo_docs/documentation_standards|Documentation standards]]"
      - "[[docs/repo_docs/ai_documentation_style|Documentation style]]"
      - "[[docs/repo_docs/content_management|Content management]]"
      - "[[docs/repo_docs/linking_validation|Link validation]]"
      - "[[docs/repo_docs/getting_started|Getting started]]"
---

# Repository documentation instructions

The `repo_docs/` folder records repository-wide documentation standards,
maintenance practices, linking rules, and the getting-started guide.

## Ground truth

- The enforceable checks are `code/scripts/validate_docs.py`,
  `verify_links.py`, `check_markdown_links.py`, `normalize_frontmatter.py`, and
  `kb_index.py`, plus `.markdownlint.json` for editor linting. The standards
  pages describe conventions; where a page and a script disagree, the script
  wins and the page is the defect.
- `getting_started.md` is the fresh-checkout guide and must match the root
  `README.md` and `pyproject.toml`.

## Rules

- Name only tools, flags, paths, and classes that exist. Check with `--help`,
  `fd`, or `rg` before writing them down.
- Do not describe documentation automation (Sphinx, MkDocs, embeddings,
  pre-commit hooks, a Makefile) that this repository does not have.
- The helper scripts in `repo_scripts/` are excluded from linting and are not
  part of the quality gate; document their real flags and no more.
- Terms listed in `docs/policy/documentation_terms.yaml` may not appear in
  tracked text files.
