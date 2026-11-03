---
title: OKN query instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `docs/okn/queries/`

- Name a graph or predicate only if the export emits it; read
  `code/scripts/export_rdf.py` before changing a query.
- `cross_graph_obo.rq` is bound to the capture in
  `docs/okn/evidence/2026-10-23-entity-model` and to the explanation. Changing it
  means capturing new evidence, not editing the old capture.
- Run `python -m scripts.export_rdf --check-submission` after a query change that
  touches submission evidence.
- Parent rules: [../AGENTS.md](../AGENTS.md).
