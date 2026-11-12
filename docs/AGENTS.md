---
title: Documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-10-13
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

- `docs/README.md` — curated navigation hub for the whole tree.
- `docs/documentation-index.md` — generated flat Markdown inventory; regenerate
  with `code/scripts/docs_index.py --write .` in the locked development environment.
  Agent instructions and the index itself are excluded.
- `docs/manuscript/README.md` — the executable end-to-end example.
- `docs/api/README.md` — API documentation policy and reference.
- `docs/cli/README.md` — the `cognitive` command (`settings`, `simulate`, `sweep`,
  `plot`, `continuous`, `hierarchical`, `benchmark`) and the report and export commands.
- `docs/visualization/README.md` — the figure gallery; its images are generated
  by the code, so regenerate them rather than editing them.
- `docs/architecture/README.md` — how the settings, simulation, plotting and
  command-line modules fit together.
- `docs/gnn.md` — the GNN declaration of the canonical discrete model, its
  binding to the canonical configuration, and the validation commands.
- `docs/config/settings.md` — the reference for every key code reads from
  `config.yaml`; the tests in `code/tests/test_docs_sync.py` require each key and
  its default to appear there.
- `examples/README.md` (repository root) — the runnable example scripts; the
  Markdown pages in `docs/examples/` explain, the scripts run.
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

From the repository root:

```bash
./run.sh docs
```

The shared root entry point runs the locked Python development environment.
The docs arm includes `validate_docs.py` (frontmatter, Python fences, public
exports, term policy, wiki and Markdown links, knowledge-base catalogs,
manuscript references), generated documentation-index and gate-evidence
freshness, and OKN submission, quality, and semantic-review checks.
See `docs/development/README.md` for index/catalog regeneration and the other
shared gate arms. Keep measured counts and timings in the generated
`docs/development/gate-measurements.md`, sourced from its JSON evidence.
