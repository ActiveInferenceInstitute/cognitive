---
title: Semantic structure of the vault
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - semantic
  - frontmatter
  - links
semantic_relations:
  - type: relates
    links:
      - "[[docs/repo_docs/ai_file_organization|File organization]]"
      - "[[docs/repo_docs/ai_validation_framework|Validation framework]]"
      - "[[docs/repo_docs/obsidian_linking|Obsidian linking]]"
---

# Semantic structure of the vault

The knowledge base is a graph. Its structure comes from three plain-text
mechanisms, all checked by scripts in `code/scripts/`. The repository does not
compute text embeddings, run semantic search, or infer relationships
automatically; there is no such code in `code/tools/src/`.

## 1. Wiki links

`[[target]]`, `[[target|display text]]`, and `[[target#heading]]` link pages.
Obsidian builds its graph view and backlinks from them. Resolution and checking
are described in [[docs/repo_docs/obsidian_linking|Obsidian linking]].

## 2. Typed relations in frontmatter

A page declares typed relations to other pages:

```yaml
semantic_relations:
  - type: implements
    links:
      - "[[knowledge_base/cognitive/active_inference|Active Inference]]"
  - type: relates
    links:
      - "[[knowledge_base/mathematics/information_theory|Information theory]]"
```

The `type` is free text. The most common values in the vault are `relates`,
`implements`, `foundation`, `related`, `extends`, `relates_to`, `organizes`, and
`documents`; prefer an existing value to a new one. The `links` must be quoted
strings. `normalize_frontmatter.py` enforces this:

```bash
python code/scripts/normalize_frontmatter.py --check .   # list drifted files, exit 1 if any
python code/scripts/normalize_frontmatter.py --write .   # rewrite them in place
```

It rewrites only the `semantic_relations` block and leaves every other key and
the body untouched. `validate_docs.py` reports an unquoted link as a
documentation error.

## 3. Generated catalogs

`kb_index.py` writes a `catalog.md` for each knowledge-base domain from the
pages' titles and descriptions (see
[[docs/repo_docs/ai_file_organization|File organization]]). A page missing from
its catalog makes `validate_docs.py` report a stale catalog.

## Querying the graph

Use the tools you already have: Obsidian's search and graph view, `rg` over the
Markdown, or a script that reads the frontmatter with `yaml.safe_load`. For
example, this lists the relation types declared on one page:

```python
import re
from pathlib import Path

import yaml

text = Path("docs/repo_docs/ai_semantic_processing.md").read_text(encoding="utf-8")
front = yaml.safe_load(re.match(r"---\n(.*?)\n---\n", text, re.S).group(1))
print([relation["type"] for relation in front["semantic_relations"]])
```

## Related documentation

- [[docs/repo_docs/ai_file_organization|File organization]]
- [[docs/repo_docs/ai_validation_framework|Validation framework]]
- [[docs/repo_docs/linking_patterns|Linking patterns]]
