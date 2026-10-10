---
title: Obsidian usage
type: guide
status: stable
created: 2024-02-12
updated: 2026-10-10
tags:
  - obsidian
  - guide
  - overview
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/documentation_standards|Documentation standards]]"
  - type: relates
    links:
      - "[[docs/repo_docs/folder_structure|Folder structure]]"
      - "[[docs/repo_docs/linking_patterns|Linking patterns]]"
---

# Obsidian usage

The repository root is an Obsidian vault. Opening the cloned folder as a vault
gives the graph view, backlinks, and wiki-link navigation over more than a thousand
Markdown files. Nothing in the Python package depends on Obsidian; the scripts
read the same files as plain text.

## Opening the vault

1. Install Obsidian from [obsidian.md](https://obsidian.md).
2. Choose "Open folder as vault" and select the repository root.
3. Trust the vault's settings when prompted. They are stored in `.obsidian/`
   and tracked in git.

## What is configured

- `app.json`: `alwaysUpdateLinks` is on, so renames inside Obsidian rewrite
  inbound links. Unsupported files are shown; deletion prompts are off.
- `core-plugins.json`: file explorer, search, quick switcher, graph, backlinks,
  outgoing links, tag pane, canvas, page preview, daily notes, templates, note
  composer, command palette, bookmarks, outline, word count, and bases are on.
  The Properties view is off.
- `graph.json`: graph view shows orphans and has no colour groups.
- `community-plugins.json`: one community plugin, `3d-graph`.

The Templates core plugin is enabled but no template folder is set in the
tracked settings. Node templates live in `docs/templates/` and are filled by
`cognitive-create-node`, not by Obsidian (see below).

## Working in the vault

- **Find a page.** Start at `knowledge_base/index.md` or a domain's generated
  `catalog.md`; use the quick switcher or search for a known title.
- **Link pages.** Write `[[folder/page|Text]]` links; see
  [[docs/repo_docs/obsidian_linking|Obsidian linking]] for how they resolve and
  what the checker enforces.
- **Create a node.** `cognitive-create-node --type agent --name my_agent` renders
  `docs/templates/agent_template.md` into `knowledge_base/agents/my_agent.md`.
  Available types are the `<type>_template.md` files in `docs/templates/`.
- **Check your work.** Run `python code/scripts/validate_docs.py --json` and
  `python code/scripts/kb_index.py --write` before committing; see
  [[docs/repo_docs/validation|Validation]].

## Troubleshooting

- A link that Obsidian shows as unresolved but the checker accepts is probably a
  bare concept name; the checker treats those as intentional.
- A link the checker reports as broken is a path or file-suffixed target that
  does not exist; correct the path.
- After renaming files outside Obsidian, run `python code/scripts/verify_links.py .`
  to find links that now point nowhere.

## Related documentation

- [[docs/repo_docs/folder_structure|Folder structure]]
- [[docs/repo_docs/linking_patterns|Linking patterns]]
- [[docs/templates/template_guide|Template guide]]
