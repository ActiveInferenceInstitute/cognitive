---
title: Obsidian usage
type: tool
status: stable
created: 2026-08-02
tags:
  - obsidian
  - vault
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../repo_docs/obsidian_linking|Obsidian linking standards]]'
      - '[[../repo_docs/linking_patterns|Linking patterns]]'
---

# Obsidian usage

The repository is an Obsidian vault: `knowledge_base/` and `docs/` use
`[[wiki links]]`, YAML frontmatter, and `semantic_relations` for graph
navigation. The `.obsidian/` folder contains the vault configuration
(`app.json`, `graph.json`, `community-plugins.json`, and others).

## Conventions

- **Frontmatter**: every page starts with YAML frontmatter (`title`, `type`,
  `status`; `created`/`updated`; `tags`; `semantic_relations`).
- **Links**: use `[[wiki links]]` for internal references. Explicit
  file-like targets must resolve; extensionless concept links are an
  intentional graph edge (see `code/scripts/verify_links.py`).
- **Graph view**: use Obsidian's graph view to explore the vault; the
  `semantic_relations` frontmatter drives typed relationships.

## Validation

```bash
python code/scripts/verify_links.py . --json
python code/scripts/check_markdown_links.py . --json
```

See `docs/repo_docs/obsidian_linking.md` and
`docs/repo_docs/linking_patterns.md` for the full standards.
