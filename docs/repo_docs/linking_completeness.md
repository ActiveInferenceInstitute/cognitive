---
title: Linking completeness
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - linking
  - completeness
  - quality
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/ai_documentation_style|Documentation style]]"
  - type: extends
    links:
      - "[[docs/repo_docs/obsidian_linking|Obsidian linking]]"
---

# Linking completeness

What a fully linked page looks like in this vault, which parts are enforced, and
how to find pages that fall short.

## Checklist for a new page

1. Frontmatter with `title`, `type`, and `status`, and at least one
   `semantic_relations` entry whose links are quoted. *Enforced:* quoting, by
   `validate_docs.py`. Having a relation at all is a convention.
2. Every explicit link (a path, or a name with a file suffix) resolves.
   *Enforced:* `verify_links.py`.
3. Standard Markdown links to files and headings resolve. *Enforced:*
   `check_markdown_links.py`.
4. The page appears in its domain's `catalog.md`. *Enforced:* `kb_index.py
   --check`; run `kb_index.py --write` after adding a page.
5. A "Related concepts" or "Related documentation" section at the end of the
   body names the closest neighbours, each with a short reason.
6. The page is linked from at least one hub (`README.md`, `index.md`, or a
   guide) so a reader can reach it without the catalog. *Convention only.*

## Finding gaps

Pages with no `semantic_relations`:

```python
import re
from pathlib import Path

import yaml

missing = []
for path in sorted(Path("knowledge_base").rglob("*.md")):
    if path.stem in {"README", "AGENTS", "catalog"}:
        continue
    match = re.match(r"---\n(.*?)\n---\n", path.read_text(encoding="utf-8"), re.S)
    front = yaml.safe_load(match.group(1)) if match else {}
    if not (front or {}).get("semantic_relations"):
        missing.append(path.as_posix())
print(len(missing), "pages without semantic_relations")
print(missing[:3])
```

Bare concept names that point at nothing: `verify_links.py . --strict-wiki-links
--json` (see [[docs/repo_docs/linking_validation|Link validation]]). A large
share of the vault's links are concept names by convention, so treat the output
as a list of pages worth writing, not as errors.

## What is not required

- No minimum number of links per page, no required link types, and no
  reciprocity. Add links that help a reader, not links that fill a quota.
- Pages are not required to link to code. Name code in code spans and link to
  the folder's `README.md` or `AGENTS.md` when the folder is the subject.

## Related documentation

- [[docs/repo_docs/linking_patterns|Linking patterns]]
- [[docs/repo_docs/linking_analysis|Linking analysis]]
- [[docs/repo_docs/linking_validation|Link validation]]
