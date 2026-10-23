---
title: Inspecting captured cross-graph assertion paths
type: reference
status: stable
updated: 2026-10-23
---

# Inspecting captured cross-graph assertion paths

The [captured result](evidence/2026-10-23-entity-model/join-results.md) connects a
document, its topic, a reviewed Wikidata mapping and returned external ontology
mapping/label assertions. The optional `daf-okn explain-join` workflow makes the supporting
predicates, review records and raw response locations inspectable for each path.
It produces a separately versioned report beside the capture.

The current offline example includes the complete
[support report](explanations/2026-10-23-entity-model/join-support.md),
[machine-readable evidence](explanations/2026-10-23-entity-model/join-support.json)
and [support manifest](explanations/2026-10-23-entity-model/support-manifest.json).
Continuous integration recomputes and checks these bytes using the pinned methods
revision and RDFLib version in the workflow below.

Use the compatible daf-okn revision pinned in
[`quality.yml`](../../.github/workflows/quality.yml). Generate the canonical
Resource Description Framework (RDF) data
first, then explain the existing capture offline:

```bash
uv run --extra dev python -m scripts.export_rdf
daf-okn explain-join --source output/okn/cognitive-kb.nq \
  --query docs/okn/queries/cross_graph_obo.rq \
  --capture docs/okn/evidence/2026-10-23-entity-model \
  --output output/okn-join-support
daf-okn explain-join --source output/okn/cognitive-kb.nq \
  --query docs/okn/queries/cross_graph_obo.rq \
  --capture docs/okn/evidence/2026-10-23-entity-model \
  --output output/okn-join-support --check
```

The workflow verifies capture custody and recomputed result bytes before
publishing support. The current explanation profile uses schema-3 captures and
complete local mapping reviews. Historical captures retain their original replay
contracts. See the methods repository's
[explanation contract](https://github.com/docxology/daf-okn/blob/main/docs/okn/join-explanations.md)
for exact supported terms, output schema, limits and failure behavior.

## Reading the evidence

Document source, mapping review source and remote observation time have distinct
meanings. The local review describes its specific subject/predicate/object;
review records for a different target do not support this path. The remote
mapping and label retain the graph, query, response hash and binding location
that supplied them.

The original string-valued rows may share several language/datatype solutions.
The explanation retains those alternatives and their multiplicity together,
without assigning an invented unique language to an old display row. Raw
response terms and any parser materialization differences remain distinguishable.

The path records each assertion separately. It performs no composed mapping
inference and adds no assertion to the canonical N-Quads. Remote labels alone
provide no review of the local concept's denotation. Deprecation/replacement
information was not requested by the captured label queries; that state remains
explicitly uncaptured. Hash agreement establishes consistency with the supplied
artifacts, while source review and scientific interpretation remain separate.

## Review work beyond a joined row

The [endpoint curation workflow](curation-workflow.md) and generated
[semantic review queue](semantic-review.md) show ambiguous source
interpretations involving document-only endpoints. Their group memberships can
overlap; distinct assertion coverage is reported separately. Source-declared
roles and relation direction remain preserved for a reviewer to inspect.

The [alignment queue](alignment-review.md) addresses external correspondence
review. A missing join, an unmapped topic and an unresolved conceptual
interpretation are separate dispositions; none authorizes an automatic mapping
or topic assignment.
