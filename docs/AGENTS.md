---
title: Documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-10-09
tags:
  - documentation
  - agents
  - framework
  - technical_writing
semantic_relations:
  - type: organizes
    links:
      - "[[README|Documentation hub]]"
      - "[[api/README|API]]"
      - "[[config/README|Configuration]]"
      - "[[development/README|Development]]"
      - "[[examples/README|Examples]]"
      - "[[guides/README|Guides]]"
      - "[[implementation/README|Implementation]]"
      - "[[research/README|Research]]"
      - "[[tools/README|Tools]]"
      - "[[templates/README|Templates]]"
      - "[[repo_docs/README|Repository documentation]]"
      - "[[manuscript/README|Manuscript]]"
      - "[[agents/README|Agent documentation]]"
      - "[[policy/README|Documentation term policy]]"
      - "[[okn/README|Open Knowledge Network export]]"
---

# Documentation instructions

This directory is the documentation tree for the repository. It mixes
executable documentation (commands, configuration, API references) with
conceptual and educational material. Keep the two clearly separated.

## Entry points

- `docs/README.md` — navigation spine for the whole tree.
- `docs/manuscript/README.md` — the executable end-to-end example.
- `docs/api/README.md` — API documentation policy and reference.
- `docs/repo_docs/getting_started.md` — fresh-checkout setup.
- `project_structure.md` (repository root) — folder map of the package and the vault.

## Authoring rules

- Describe only behavior that exists in this repository. Runtime examples must
  import from the installed `cognitive` or `Things` packages.
- Conceptual pages may explain theory and domain context, but must not present
  illustrative classes, commands, or modules as available interfaces.
- Keep YAML frontmatter valid; use wiki links for cross-references. Do not
  invent file paths, citation keys, statistics, or external URLs.
- When a page documents a command, run it first and record real output.
- Quote `semantic_relations` links (`"[[page]]"`); unquoted links are rejected.
- Avoid the words listed in `docs/policy/documentation_terms.yaml`; the gate
  rejects them anywhere in tracked text.
- Run a changed Python example before keeping it; the gate only compiles it.

## Validation

```bash
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
python code/scripts/normalize_frontmatter.py --check .
python code/scripts/kb_index.py --check .
```

The first is the aggregate gate (frontmatter, Python fences, public exports,
term policy, wiki and Markdown links, knowledge-base catalogs, manuscript
references). The second checks explicit file-like wiki links. The last two are
also run by the first; each accepts `--write` to apply its fix. See
`docs/development/README.md` for the full development loop.
