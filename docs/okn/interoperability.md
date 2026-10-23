---
title: Cognitive graph interoperability
type: reference
status: stable
updated: 2026-10-23
---

# Cognitive graph interoperability

The graph describes explanatory pages and concepts. Reusing a vocabulary does
not by itself align their subjects to another graph. Existing external links
provide two concrete join paths; current coverage is generated in
[statistics](statistics.md).

| Source | Exported identifier and predicate | Intended join |
|---|---|---|
| Source-reviewed `external_mappings` | Explicit SKOS predicate from a topic concept to the reviewed target IRI | Concept correspondence to Wikidata or domain ontology terms |
| Unreviewed `wikidata: Q…` | `rdfs:seeAlso` | Discovery only; excluded from semantic joins |
| Page tags | `schema:keywords` to external terms or declared local keyword concepts | Shared subject identifiers without typing a publication as its subject |
| Citation page `doi` | `schema:sameAs` to `https://doi.org/…` | Join publication records using the same normalized DOI IRI |
| Markdown document | Stable `…/page/<path>` CreativeWork IRI | Source metadata, citations and document references |
| Topic concept | Separate `…/concept/<path>` SKOS Concept IRI | Subject of the page, connected through `schema:about` and `dcterms:subject`; conceptual relations and reviewed mappings |

The [vocabulary mapping](vocabulary.md) documents the source-review contract and
its concept-level interpretation. The [generated priority queue](alignment-review.md)
ranks incoming references and lists current reviewed assertions. Unreviewed
Wikidata proposals are preserved as discovery links, rather than published as
close matches. Exact matches require compatible definitions and scope.

The [entity model](entity-model.md) separates documents from what they describe.
Mapping queries retrieve a concept's source through its describing page, rather
than treating that page as the concept. Keywords also describe document subjects;
the [keyword review](keyword-curation-2026-10-23.md) records external term reuse
and scope-based deferrals. Neither a subject link nor a keyword makes the document
an instance of its subject.

## Runnable query evidence

[Concept mappings](queries/concept_mappings.rq) lists concept IRIs, labels,
external IRIs, predicates and source pages. [Mapping evidence](queries/mapping_evidence.rq)
retrieves sources, rationales and review dates; [keyword concepts](queries/keyword_concepts.rq)
retrieves concept-valued tags. [Citation identifiers](queries/citation_identifiers.rq)
lists DOI join keys and source pages. These queries use the actual named graph.
The regression suite parses the exported N-Quads with RDFLib and executes these
queries against the repository graph, requiring nonempty results.

```bash
uv run --extra dev python -m pytest code/tests/test_export_rdf.py -q
```

The [captured cross-graph query](queries/cross_graph_obo.rq) joins those keys to
the OKN identifier-mappings graph and then to OBO labels in Ubergraph.
[Generated results and linked source pages](evidence/2026-10-23-entity-model/join-results.md)
record the actual live-response join. Raw responses, queries and hashes are
preserved beside that report. The join was executed locally against cognitive's
export and live OKN responses; cognitive-kb is not yet hosted in the federation.

Capture/replay and submission methods live in
[daf-okn](https://github.com/docxology/daf-okn), not in this graph repository.
With the methods installed, replay the captured response files offline:

```bash
daf-okn join --source output/okn/cognitive-kb.nq \
  --query docs/okn/queries/cross_graph_obo.rq \
  --replay docs/okn/evidence/2026-10-23-entity-model --output output/okn-replay
```

The cross-graph query is ready for the server-side federation after cognitive-kb
is deployed. Until then, a server-only query with its local graph clause returns
no cognitive pages. Reviewed mappings retain their stated predicate after a successful
join; shared identifiers and ontology labels do not validate scientific claims.
Historical [October 13 results](evidence/2026-10-13/join-results.md) preserve the
earlier discovery experiment, including obsolete labels. Its manifest binds the
old export and query hashes and cannot replay against the revised export. It is
not current mapping evidence. Fresh response labels likewise do not justify
copying a discovered term into the reviewed mapping set.

The [earlier October 20 capture](evidence/2026-10-20/join-results.md) and
[October 20 graph-methods capture](evidence/2026-10-20-graph-methods/join-results.md)
are preserved with their original export/query hashes and original combined
page/concept model. They cannot replay against this revised export and query.
The current capture binds the stored join query, result bytes and structured
provenance. Replay checks the complete stage and file inventory.

The optional [join-support workflow](join-support.md) exposes the precise local
mapping predicate and review records beside each captured remote assertion.
Typed solutions and raw response origins remain visible, including alternatives
that share an old string-valued row. Its separate report preserves all existing
capture bytes and leaves the submitted RDF unchanged. The
[endpoint review queue](semantic-review.md) serves a separate source-role
curation task; ordinary document relations retain their existing interpretation.

## Cross-graph signposts

| Target | Role and identifier path | Evidence |
|---|---|---|
| [Wikidata](https://registry.okn.us/registry/kgs/wikidata/) | Reviewed mappings plus separate discovery links | Explicit SKOS records; `rdfs:seeAlso` is excluded from joins |
| [Identifier mappings](https://registry.okn.us/registry/kgs/identifier-mappings/) | Wikidata IRI to OBO IRI through the target graph's `skos:exactMatch` | Captured bounded queries and responses |
| [Ubergraph](https://registry.okn.us/registry/kgs/ubergraph/) | Labels for those OBO identifiers in the hosted ontology graph | Captured label responses and joined result rows |

The exported dataset includes `rdfs:seeAlso` links to these registry entries.
The generated report links each matched concept page, Wikidata key and ontology
term. Imported mapping and label evidence remains attributed to its named source
graph; the submitted cognitive N-Quads does not include those remote triples.

## Domain alignment review

Jim's examples of MONDO and KWG identifiers are domain-specific. A page about a
disease should only be mapped to MONDO after its denotation is checked; a page
about geospatial theory is not a geographic feature and should not receive a KWG
feature identifier. The exporter accepts reviewed HTTP(S) domain identifiers and
publication DOIs. It does not infer MONDO or KWG mappings from keywords.

For a new mapping, record the source page, target identifier, intended predicate,
target definition, source definition, reviewer, review date and evidence in the
change description. Check identifier spelling, denotation, mapping direction and
the target vocabulary's intended use. Add export support and rejection tests
together with its vocabulary documentation. For additional target graphs, inspect
their identifier conventions and record actual joins, including empty or ambiguous
results. The captured OBO path demonstrates shared-identifier discovery; it does
not replace a review of each page's denotation.
