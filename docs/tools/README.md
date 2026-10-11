---
title: Tools documentation index
type: documentation
status: stable
updated: 2026-10-10
semantic_relations:
  - type: organizes
    links:
      - "[[docs/tools/AGENTS|Tools documentation instructions]]"
      - "[[docs/development/README|Development resources]]"
      - "[[docs/tools/doc_validator|Documentation validator]]"
---

# Tools documentation index

This directory documents the tooling the repository actually ships: the
installed CLI entry points, the validation scripts, and the build tooling.
It is not a catalog of hypothetical tools.

## Executable tooling

Installed console commands (from `pyproject.toml`, available after
`python -m pip install -e ".[dev]"`):

- `cognitive-create-node` — create knowledge nodes.
- `cognitive-verify-links` — validate Obsidian wiki links.
- `cognitive-validate-docs` — validate documentation, examples, and links.
- `cognitive-normalize-frontmatter` — check or rewrite `semantic_relations`
  links as quoted strings (`--check` / `--write`).
- `cognitive-kb-index` — check or regenerate the knowledge-base catalogs
  (`--check` / `--write`).
- `cognitive-benchmark` — run the runtime benchmarks.
- `cognitive-build-manuscript` — build the executable manuscript.
- `cognitive-mutation-check` — run the mutation gate.

Repository scripts (run with `python code/scripts/...`):

- `validate_docs.py` — the documentation gate (frontmatter, Python fenced
  blocks, exports, wiki links, standard Markdown links, manuscript).
- `verify_links.py` — explicit wiki-link validation.
- `check_markdown_links.py` — standard Markdown link and anchor validation.
- `normalize_frontmatter.py` — `semantic_relations` quoting.
- `kb_index.py` — knowledge-base catalog generation.
- `build_manuscript.py` — manuscript build pipeline.
- `mutation_check.py` — mutation gate.

Static and test tooling is `ruff`, `mypy`, `pytest`, and `coverage`, configured
in `pyproject.toml` and run by `.github/workflows/quality.yml`.

Helper scripts for documentation maintenance live in
`docs/repo_docs/repo_scripts/`.

## Related pages

- Topic pages (for example `link_checker.md`, `benchmark_tools.md`,
  `mutation_testing.md`, `plotting_tools.md`) document the tooling for each
  topic. `dependency_analyzer.md`, `static_analyzer.md`, `graph_generator.md`,
  `test_generators.md`, and `model_context_protocol.md` are short pages stating what the repository
  uses in place of a dedicated tool of that name (or that it has none).
- `git_tools.md` and `git_workflow.md` describe plain Git use here.
- See `AGENTS.md` for the accuracy rules.
