---
title: Link checker
type: tool
status: stable
updated: 2026-10-10
tags:
  - links
  - tools
semantic_relations:
  - type: documents
    links:
      - "[[docs/tools/advanced_linking|Advanced linking]]"
      - "[[docs/tools/doc_validator|Documentation validator]]"
---

# Link checker

`code/scripts/verify_links.py` validates Obsidian wiki links without treating
every unresolved concept as a broken file reference.

Default mode resolves existing wiki targets and reports only explicit file
references that are missing, such as `[[missing.md]]`. Extensionless links such
as `[[active_inference]]` or `[[systems/network_theory]]` are intentional graph
edges in the knowledge base and are skipped when no matching file exists.

Run the default gate:

```bash
python code/scripts/verify_links.py .
```

Run a stricter audit when you want every wiki link to resolve to a file:

```bash
python code/scripts/verify_links.py . --strict-wiki-links
```

Add `--json` for the complete machine-readable report. The installed command is
`cognitive-verify-links`. Standard Markdown links and heading anchors are
checked separately by `code/scripts/check_markdown_links.py`.
