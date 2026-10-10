---
title: File organization for human and AI readers
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - organization
  - structure
  - metadata
semantic_relations:
  - type: relates
    links:
      - "[[docs/repo_docs/ai_documentation_style|Documentation style]]"
      - "[[docs/repo_docs/folder_structure|Folder structure]]"
      - "[[docs/repo_docs/naming_conventions|Naming conventions]]"
      - "[[docs/repo_docs/ai_semantic_processing|Semantic structure]]"
---

# File organization for human and AI readers

The repository has no separate AI-only layout. The same files serve people and
agents, and four conventions make them machine-navigable.

## 1. A README and an AGENTS file per folder

`README.md` is the overview and navigation page for a folder. `AGENTS.md` holds
the technical notes and rules for anyone (human or agent) changing that
folder. The root [`AGENTS.md`](../../AGENTS.md) and
[`CLAUDE.md`](../../CLAUDE.md) are the entry points; the root
[`project_structure.md`](../../project_structure.md) maps the directories.
Most folders carry both files; a few leaf folders carry one or neither.

## 2. Frontmatter on every page

Frontmatter gives a page a `title`, a `type`, a `status`, dates, and typed
`semantic_relations` to its neighbours. Fields and rules are in
[[docs/repo_docs/documentation_standards|Documentation standards]]. The
frontmatter is the metadata; there are no in-body metadata markers, and no
per-directory metadata files.

## 3. Predictable names

Lowercase `snake_case` Markdown files, with `README.md` and `AGENTS.md` as the
capitalized exceptions. See [[docs/repo_docs/naming_conventions|Naming conventions]].

## 4. Generated catalogs

Each `knowledge_base/` domain has a `catalog.md` listing every page under it
with its title and one-line description, plus `knowledge_base/catalog.md` for
the domains. They are produced by `code/scripts/kb_index.py`:

```bash
python code/scripts/kb_index.py --check .   # exit 1 if any catalog is stale
python code/scripts/kb_index.py --write .   # regenerate stale catalogs
```

A catalog is derived data. To change an entry, edit the page's `title` or
`description` frontmatter (or its opening paragraph) and regenerate.
`README.md`, `AGENTS.md`, and `catalog.md` files are not catalogued.

## Templates

New nodes start from the files in `docs/templates/`. `cognitive-create-node`
renders `<type>_template.md` for a node type; see
[[docs/templates/template_guide|Template guide]].

## Related documentation

- [[docs/repo_docs/folder_structure|Folder structure]]
- [[docs/repo_docs/ai_folder_structure|AI-oriented folder notes]]
- [[docs/repo_docs/ai_semantic_processing|Semantic structure]]
