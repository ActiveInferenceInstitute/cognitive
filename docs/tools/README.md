---
title: Tools documentation index
type: documentation
status: stable
semantic_relations:
  - type: organizes
    links:
      - '[[AGENTS|Tools documentation instructions]]'
      - '[[development_tools_index|Development tools index]]'
      - '[[../../code/scripts/validate_docs.py|Documentation validator]]'
---

# Tools documentation index

This directory documents the tooling the repository actually ships: the
installed CLI entry points, the validation scripts, and the build tooling.
It is not a catalog of hypothetical tools.

## Executable tooling

Installed console commands (from `pyproject.toml`, available after
`python -m pip install -e ".[dev]"`):

- `cognitive-create-node` — create knowledge nodes from YAML.
- `cognitive-verify-links` — validate Obsidian wiki links.
- `cognitive-validate-docs` — validate documentation, examples, and links.
- `cognitive-benchmark` — run the runtime benchmarks.
- `cognitive-build-manuscript` — build the executable manuscript.

Repository scripts (run with `python code/scripts/...`):

- `validate_docs.py` — the documentation gate (frontmatter, Python fenced
  blocks, exports, wiki links, standard Markdown links, manuscript).
- `verify_links.py` — explicit wiki-link validation.
- `check_markdown_links.py` — standard Markdown link and anchor validation.
- `build_manuscript.py` — manuscript build pipeline.

Helper scripts for documentation maintenance live in
`docs/repo_docs/repo_scripts/`.

## Related pages

- `development_tools_index.md` and `development_tools.md` — older tool
  documentation pages; treat them as conceptual unless the tool they name
  exists in `code/scripts/` or `pyproject.toml`.
- Individual topic pages (for example `git_tools.md`) may describe
  speculative tooling; see `AGENTS.md` for the accuracy rules.
