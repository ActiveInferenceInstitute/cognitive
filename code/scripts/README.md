---
title: Repository scripts
type: documentation
status: stable
updated: 2026-10-13
semantic_relations:
  - type: documents
    links:
      - "[[AGENTS|Script instructions]]"
      - "[[docs/development/README|Development]]"
      - "[[docs/visualization/gallery|Gallery]]"
---

# Repository scripts

The installed `scripts` package contains documentation, graph, manuscript and
verification commands. Console entry points are declared in `pyproject.toml`.

| Module | Purpose |
|---|---|
| `validate_docs` | Aggregate documentation gate. |
| `verify_links`, `check_markdown_links`, `markdown_fences` | Wiki links, Markdown links and Python fences. |
| `normalize_frontmatter`, `kb_index` | Frontmatter and knowledge-base catalogs. |
| `export_rdf`, `kb_report` | RDF export and graph measurements. |
| `build_manuscript` | Source-driven manuscript artifacts and receipts. |
| `mutation_check` | Source mutations that the suite must detect. |
| `regenerate_gallery` | Reproduce all gallery figures, their captions and hash receipts. |

```bash
uv run python -m scripts.regenerate_gallery --publish
uv run python -m scripts.validate_docs --json
```

See [[docs/development/README|development]] for the complete gate sequence.
