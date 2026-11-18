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

For `[[target]]` in a file, `verify_links.py` looks for a file relative to the
linking file or the repository root. It tries the given path, then the same
name with `.md` when the suffix is omitted, then `README.md` and `index.md`
inside a matching directory. Exact spelling takes precedence over
case-insensitive candidates.

For a target without a recognized file suffix, the checker falls back to a
matching Markdown basename, then to a frontmatter `aliases` entry. If more
than one file has the same basename, the first in scan order wins. A target
can resolve this way even when its directory component is misspelled.
Alias matching ignores case and treats hyphens and underscores as spaces.

For a link with `#heading`, the checker compares the normalized fragment with
the headings in the resolved file. A missing heading produces an advisory
warning.

## Concept links versus file links

Use a source-relative or repository-root-relative path for an explicit file
reference. Include the `.md` suffix on a Markdown path when a missing or
misspelled path should fail the check without a basename or alias lookup.
An unresolved path or recognized file name is reported as broken. An unresolved
bare name is counted as a skipped concept link in the default mode.

Use `verify_links.py --strict-wiki-links` to report unresolved bare names as
well. This option keeps the same resolution rules. The documentation gate uses
the default mode.

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
