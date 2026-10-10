---
title: Documentation tools
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/tools/doc_validator]]"
      - "[[docs/tools/link_checker]]"
      - "[[docs/policy/README]]"
---

# Documentation tools

Console commands declared in `pyproject.toml`:

| Command | Purpose |
|---|---|
| `cognitive-validate-docs` | frontmatter, structure, forbidden terms, catalog freshness |
| `cognitive-verify-links` | wiki-link resolution |
| `cognitive-normalize-frontmatter` | rewrites `semantic_relations` into quoted wiki links |
| `cognitive-kb-index` | regenerates the knowledge-base catalogs |
| `cognitive-create-node` | renders a knowledge-base node from a template |
| `cognitive-build-manuscript` | builds the manuscript |

See [[docs/tools/doc_validator|the validator page]],
[[docs/tools/link_checker|the link checker page]], and the term policy in
[[docs/policy/README|the policy docs]].
