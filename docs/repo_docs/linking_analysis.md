---
title: Linking analysis
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - linking
  - analysis
  - documentation
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/obsidian_linking|Obsidian linking]]"
  - type: extends
    links:
      - "[[docs/repo_docs/ai_validation_framework|Validation framework]]"
---

# Linking analysis

How to measure the state of the vault's links with what the repository already
provides. There are no stored link-quality scores or threshold files; every
number comes from running a script, so it is current by construction.

## Link health

`verify_links.py --json` is the source of link counts. This script reports the
share of links that resolve to a page versus those that are bare concept names:

```python
import json
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "code/scripts/verify_links.py", ".", "--json"],
    capture_output=True,
    text=True,
)
report = json.loads(result.stdout)
concept_share = report["skipped_concept_links"] / report["checked_links"]
print(f"{report['checked_links']} links, {concept_share:.0%} are concept links")
print(f"{len(report['broken_links'])} broken, {len(report['anchor_warnings'])} anchor warnings")
```

Run it from the repository root. At the time of writing, about a third of the
vault's wiki links are concept links and none are broken. The concept share is
the number to watch: it is the part of the graph the gate does not protect. To
list the unresolved bare names, run `verify_links.py . --strict-wiki-links
--json` and read `broken_links`.

## Relation types

The typed relations in frontmatter can be tallied directly from the YAML:

```python
import re
from collections import Counter
from pathlib import Path

import yaml

types = Counter()
for path in Path("knowledge_base").rglob("*.md"):
    match = re.match(r"---\n(.*?)\n---\n", path.read_text(encoding="utf-8"), re.S)
    front = (yaml.safe_load(match.group(1)) if match else None) or {}
    for relation in front.get("semantic_relations") or []:
        if isinstance(relation, dict):
            types[relation.get("type")] += 1
print(types.most_common(5))
```

The most common types are `relates`, `implements`, `foundation`, `related`, and
`extends`. `relates` and `related` overlap, as do `foundation` and
`foundation_for`; consolidating them would change many files and is not done
by any script.

## What the graph does not tell you

- **Orphans.** Obsidian's graph view (`showOrphans` is on in
  `.obsidian/graph.json`) shows pages with no links. No script reports them; the
  generated `catalog.md` files are what guarantee every knowledge-base page is
  listed somewhere.
- **Link meaning.** A resolved link says the target exists, not that it is the
  right page. Reading is the only check.
- **External URLs.** They are not fetched by any checker.

## Related documentation

- [[docs/repo_docs/linking_validation|Link validation]]
- [[docs/repo_docs/linking_completeness|Linking completeness]]
- [[docs/repo_docs/ai_semantic_processing|Semantic structure]]
