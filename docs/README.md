---
title: Documentation hub
type: documentation
status: stable
updated: 2026-10-09
semantic_relations:
  - type: organizes
    links:
      - "[[../README|Repository README]]"
      - "[[api/README|API]]"
      - "[[config/README|Configuration]]"
      - "[[development/README|Development]]"
      - "[[examples/README|Examples]]"
      - "[[guides/README|Guides]]"
      - "[[implementation/README|Implementation]]"
      - "[[research/README|Research]]"
      - "[[tools/README|Tools]]"
      - "[[templates/README|Templates]]"
      - "[[agents/README|Agent documentation]]"
      - "[[manuscript/README|Manuscript]]"
      - "[[policy/README|Documentation term policy]]"
      - "[[repo_docs/README|Repository documentation]]"
---

# Documentation hub

This directory contains documentation for the implemented Python package, the
executable manuscript, conceptual research notes, and the repository's Obsidian
knowledge-vault material. The root README and the manuscript are the canonical
working entry points.

## Start here

1. [`../README.md`](../README.md): install, a minimal discrete model, commands,
   and quality gates.
2. [`repo_docs/getting_started.md`](repo_docs/getting_started.md): fresh-checkout
   setup and verification.
3. [`manuscript/README.md`](manuscript/README.md): end-to-end build and
   reproducibility contract.
4. [`api/README.md`](api/README.md): API documentation policy.
5. [`examples/README.md`](examples/README.md): executable examples.

## Documentation domains

- [`api/`](api/README.md): public API reference, executed examples, versioning.
- [`config/`](config/README.md): the root `config.yaml` and which keys code reads.
- [`development/`](development/README.md): development loop and CI.
- [`agents/`](agents/README.md): navigation for agent documentation.
- [`manuscript/`](manuscript/README.md): the executable manuscript.
- [`policy/`](policy/README.md): the term policy the documentation gate enforces.
- [`examples/`](examples/README.md): validated example entry points.
- [`guides/`](guides/README.md): application notes and learning paths.
- [`implementation/`](implementation/README.md): implementation patterns and the
  RxInfer documentation subtree.
- [`research/`](research/README.md): research methods and application notes.
- [`repo_docs/`](repo_docs/README.md): repository documentation standards and
  maintenance material.
- [`templates/`](templates/README.md): document templates.
- [`tools/`](tools/README.md): repository tool documentation.

## Knowledge base

The conceptual knowledge base is at [`../knowledge_base/README.md`](../knowledge_base/README.md).
Its source files use Obsidian wiki links and may contain theoretical or
illustrative material beyond the implemented runtime. For code behavior, prefer
the installed package, its tests, and the manuscript's reproducibility section.

The [project structure](../project_structure.md) maps these folders; open work is
in the backlog file at the repository root.

## Validation

From the repository root:

```bash
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
python code/scripts/check_markdown_links.py . --json
python code/scripts/normalize_frontmatter.py --check .
python code/scripts/kb_index.py --check .
```

`validate_docs.py` is the aggregate gate: frontmatter, quoted
`semantic_relations` links, compile-checking of Python fences, package exports,
the term policy, wiki and Markdown links, knowledge-base catalogs, and
manuscript references. `verify_links.py` checks wiki links that name a file;
unresolved extensionless concept links are reported separately unless
`--strict-wiki-links` is passed. `normalize_frontmatter.py` and `kb_index.py` are
also run inside `validate_docs.py`; run them directly with `--write` to fix what
they report.

## Licensing

Code is MIT licensed under [`../LICENSE`](../LICENSE). Documentation and
knowledge-base pages carry CC BY-NC-SA 4.0 notices where applicable.
