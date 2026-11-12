---
title: Repository scripts
type: documentation
status: stable
updated: 2026-11-12
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
The root `./run.sh ARM` invokes `gate.py` for the same arms used by CI.

| Module | Purpose |
|---|---|
| `gate` | Independently selectable lint, types, fast/full tests, docs, GNN and manuscript arms. |
| `docs_index` | Generate or check the flat documentation index. |
| `gate_evidence` | Generate or check the local measurement document from saved records. |
| `validate_docs` | Aggregate documentation gate. |
| `verify_links`, `check_markdown_links`, `markdown_fences` | Wiki links, Markdown links and Python fences. |
| `normalize_frontmatter`, `kb_index` | Frontmatter and knowledge-base catalogs. |
| `export_rdf`, `kb_report` | RDF export and graph measurements. |
| `build_manuscript` | Source-driven manuscript artifacts and receipts. |
| `mutation_check` | Source mutations that the suite must detect. |
| `regenerate_gallery` | Reproduce all gallery figures, their captions and hash receipts. |

```bash
uv run python -m scripts.regenerate_gallery --publish
./run.sh docs
```

See [[docs/development/README|development]] for the complete gate sequence.
