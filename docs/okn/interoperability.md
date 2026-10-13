---
title: Cognitive graph interoperability
type: reference
status: stable
updated: 2026-10-13
---

# Cognitive graph interoperability

The graph describes explanatory pages and concepts. Reusing a vocabulary does
not by itself align their subjects to another graph. Existing external links
provide two concrete join paths; current coverage is generated in
[statistics](statistics.md).

| Source | Exported identifier and predicate | Intended join |
|---|---|---|
| Concept page `wikidata: Q…` | `skos:closeMatch` to `http://www.wikidata.org/entity/Q…` | Follow a candidate concept mapping to a graph using the same Wikidata IRI |
| Citation page `doi` | `schema:sameAs` to `https://doi.org/…` | Join publication records using the same normalized DOI IRI |
| Local concept/page | Stable path-based FRINK IRI | Reference a page within this graph; not an external identifier alignment |

The [vocabulary mapping](vocabulary.md) documents how the current Wikidata links
were screened. They have not each had human review. Preserve `skos:closeMatch`;
do not promote these links to `owl:sameAs` or `skos:exactMatch` based on labels alone.

## Runnable query evidence

[Concept mappings](queries/concept_mappings.rq) lists concept IRIs, labels,
Wikidata IRIs and source pages. [Citation identifiers](queries/citation_identifiers.rq)
lists DOI join keys and source pages. Both queries use the actual named graph.
The regression suite parses the exported N-Quads with RDFLib and executes these
queries against the repository graph, requiring nonempty results.

```bash
uv run --extra dev python -m pytest code/tests/test_export_rdf.py -q
```

The [captured cross-graph query](queries/cross_graph_obo.rq) joins those keys to
the OKN identifier-mappings graph and then to OBO labels in Ubergraph.
[Generated results and linked source pages](evidence/2026-10-13/join-results.md)
record the actual live-response join. Raw responses, queries and hashes are
preserved beside that report. The join was executed locally against cognitive's
export and live OKN responses; cognitive-kb is not yet hosted in the federation.

Capture/replay and submission methods live in
[daf-okn](https://github.com/docxology/daf-okn), not in this graph repository.
With the methods installed, replay the captured response files offline:

```bash
daf-okn join --source output/okn/cognitive-kb.nq \
  --query docs/okn/queries/cross_graph_obo.rq \
  --replay docs/okn/evidence/2026-10-13 --output output/okn-replay
```

The cross-graph query is ready for the server-side federation after cognitive-kb
is deployed. Until then, a server-only query with its local graph clause returns
no cognitive pages. Candidate page mappings remain candidates after a successful
join; shared identifiers and ontology labels do not validate scientific claims.
The captured labels include terms marked obsolete. The report preserves those
source responses; review target definitions, deprecation and replacements before
using a discovered correspondence as a curated mapping.

## Cross-graph signposts

| Target | Role and identifier path | Evidence |
|---|---|---|
| [Wikidata](https://registry.okn.us/registry/kgs/wikidata/) | Concept discovery through existing Wikidata IRIs | Existing page `skos:closeMatch` links |
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
feature identifier. The current exporter supports Wikidata identifiers and
publication DOIs; it does not export MONDO or KWG mappings.

For a new mapping, record the source page, target identifier, intended predicate,
target definition, source definition, reviewer, review date and evidence in the
change description. Check identifier spelling, denotation, mapping direction and
the target vocabulary's intended use. Add export support and rejection tests
together with its vocabulary documentation. For additional target graphs, inspect
their identifier conventions and record actual joins, including empty or ambiguous
results. The captured OBO path demonstrates shared-identifier discovery; it does
not replace a review of each page's denotation.
