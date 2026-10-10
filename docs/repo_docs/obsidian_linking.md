---
title: Obsidian linking
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - obsidian
  - linking
  - standards
semantic_relations:
  - type: relates
    links:
      - "[[docs/repo_docs/obsidian_usage|Obsidian usage]]"
      - "[[docs/repo_docs/linking_patterns|Linking patterns]]"
      - "[[docs/repo_docs/linking_validation|Link validation]]"
      - "[[knowledge_base/linking_standards|Knowledge base linking standards]]"
---

# Obsidian linking

How wiki links are written in this vault and how `code/scripts/verify_links.py`
resolves them. Example links below sit in code spans because the checker ignores
code; a bare link would be checked.

## Syntax

| Form | Meaning |
|---|---|
| `[[target]]` | Link to a page by name or path |
| `[[target\|text]]` | Same, shown as `text` (write the pipe as `\|` inside a table) |
| `[[target#Heading]]` | Link to a heading in the page |

Block references (`[[target#^id]]`) are an Obsidian feature, but the checker
treats the part after `#` as a heading and reports an anchor warning, so avoid
them.

## How a target resolves

For `[[target]]` in a file, `verify_links.py` tries, in order:

1. The path relative to the linking file, then relative to the repository root,
   each as given, with `README.md` or `index.md` appended, and with `.md` added
   when the target has no recognized file suffix.
2. The path as a case-insensitive repository-relative key.
3. For a target without a file suffix, a page with that path (without `.md`) or,
   failing that, any page whose file name (stem) matches. The first match in
   scan order wins.

A target that resolves is counted; if it has a `#heading`, the heading must
slugify to a heading in the target file, otherwise an anchor warning is
reported (a warning, not a failure).

## Concept links versus file links

A target that does not resolve is treated as follows:

- If it contains a `/`, ends in a recognized suffix (`.md`, `.py`, `.yaml`,
  `.png`, and so on), the link is an explicit file reference and is **broken**.
- If it is a bare name with no `/` and no suffix, it is a **concept link**: it
  is skipped and counted, on the convention that the page may be written later.
  Roughly a third of the vault's links are concept links.

`verify_links.py --strict-wiki-links` removes the second rule and reports
every unresolved target. CI uses the default mode. Use the path form
(`knowledge_base/cognitive/active_inference`) for any link that must keep
working, since a typo in a path fails the gate while a typo in a bare name does
not.

## What the checker ignores

- Fenced code blocks and inline code.
- Targets made only of digits, commas, dots, spaces, and brackets, so matrix
  literals such as `[[0.9, 0.1], [0.1, 0.9]]` are not read as links.
- The directories `.git`, `.venv`, `venv`, `node_modules`, `__pycache__`, the
  tool caches, and any directory named `output`, `Output`, or `outputs`.

## Frontmatter links

Links in `semantic_relations` must be quoted strings, for example
`- "[[knowledge_base/cognitive/active_inference|Active Inference]]"`. An unquoted
link parses as a nested YAML list and Obsidian stops seeing it as a link.
`normalize_frontmatter.py` checks and repairs this (see
[[docs/repo_docs/ai_semantic_processing|Semantic structure]]).

## Obsidian settings

`.obsidian/app.json` sets `alwaysUpdateLinks: true`, so renaming a page in
Obsidian rewrites links to it. Renaming outside Obsidian (or with `git mv`)
does not; run `verify_links.py` afterwards. Obsidian's graph and backlink views
are configured in `.obsidian/graph.json` and `.obsidian/backlink.json`.

## Related documentation

- [[docs/repo_docs/obsidian_usage|Obsidian usage]]
- [[docs/repo_docs/linking_patterns|Linking patterns]]
- [[docs/repo_docs/linking_validation|Link validation]]
- [[knowledge_base/linking_standards|Knowledge base linking standards]]
