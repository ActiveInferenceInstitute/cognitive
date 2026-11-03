---
title: OKN SPARQL queries
type: documentation
status: stable
updated: 2026-11-03
---

# `docs/okn/queries/`

Seven SPARQL queries used for the Open Knowledge Network export and its evidence:

| Query | Role |
|---|---|
| `cross_graph_obo.rq` | The cross-graph join replayed and explained by the `okn-replay` CI job. |
| `identifier_mappings.rq` | Identifier-mapping captures. |
| `citation_identifiers.rq` | Citation identifier join. |
| `concept_mappings.rq` | Concept correspondence join. |
| `document_topics.rq` | Document topic join. |
| `keyword_concepts.rq` | Keyword concept join. |
| `relation_scopes.rq` | Relation scope join. |

The exported graph is described in [`../README.md`](../README.md); the rules for
changing a query are in [`../AGENTS.md`](../AGENTS.md).
