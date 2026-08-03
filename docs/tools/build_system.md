---
title: Build system
type: tool
status: stable
created: 2026-08-02
tags:
  - build
  - manuscript
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../manuscript/README|Manuscript build]]'
---

# Build system

The repository's build tooling:

- **Manuscript build** — `cognitive-build-manuscript --output build/manuscript`
  (backed by `code/scripts/build_manuscript.py`) renders the executable
  manuscript: deterministic figures, auto-numbered equations and tables,
  Pandoc citations, HTML, and optionally a XeLaTeX PDF. Build output is
  ignored by Git; use `--no-pdf` without a LaTeX installation.
- **Package build** — `pyproject.toml` declares the setuptools configuration
  and the installed CLI entry points. Install with
  `python -m pip install -e ".[dev]"`.

See `docs/manuscript/README.md` for the full build contract.
