---
title: Documentation hub
type: documentation
status: stable
updated: 2026-10-13
semantic_relations:
  - type: organizes
    links:
      - "[[../README|Repository README]]"
      - "[[api/README|API]]"
      - "[[cli/README|Command line]]"
      - "[[visualization/README|Visualization]]"
      - "[[architecture/README|Architecture]]"
      - "[[config/README|Configuration]]"
      - "[[config/settings|Settings reference]]"
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
      - "[[okn/README|Open Knowledge Network export]]"
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
5. [`examples/README.md`](examples/README.md): example pages, with a pointer to
   the runnable scripts in [`../examples/README.md`](../examples/README.md).
6. [`cli/README.md`](cli/README.md): run a simulation or a sweep from the shell.

## Documentation domains

- [`api/`](api/README.md): public API reference, executed examples, versioning.
- [`cli/`](cli/README.md): the `cognitive` command and the report and export commands.
- [`visualization/`](visualization/README.md): the figure gallery and what each plot shows.
- [`architecture/`](architecture/README.md): how settings, simulation, plots and the command line fit together.
- [`config/`](config/README.md): the root `config.yaml` and which keys code reads;
  [`config/settings.md`](config/settings.md) is the reference for every setting.
- [`development/`](development/README.md): development loop and CI.
- [`agents/`](agents/README.md): navigation for agent documentation.
- [`manuscript/`](manuscript/README.md): the executable manuscript.
- [`policy/`](policy/README.md): the term policy the documentation gate enforces.
- [`okn/`](okn/README.md): the knowledge base as RDF for the Open Knowledge Network.
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

Code is MIT licensed under [`../LICENSE`](../LICENSE). Repository-owned knowledge-base
content and graph exports are CC BY 4.0 under [`../LICENSE-KNOWLEDGE-BASE.md`](../LICENSE-KNOWLEDGE-BASE.md).
Other documentation remains CC BY-NC-SA 4.0.
