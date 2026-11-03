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
| `mapping_evidence.rq` | Page, concept, mapping, target, evidence, rationale and review status. |
| `citation_identifiers.rq` | Page identifiers, titles, DOIs and sources. |
| `concept_mappings.rq` | Page, concept, mapping, external identifier and source. |
| `document_topics.rq` | Page, title, concept and label. |
| `keyword_concepts.rq` | Page, keyword and label. |
| `relation_scopes.rq` | Relation, source name, domain, range and meaning. |

The exported graph is described in [`../README.md`](../README.md); the rules for
changing a query are in [`../AGENTS.md`](../AGENTS.md).
