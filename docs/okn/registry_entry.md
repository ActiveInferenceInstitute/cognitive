---
title: OKN registry entry for cognitive-kb
description: The accepted cognitive-kb registry entry, its proposed update, and the handoff to lakeFS.
type: guide
status: stable
updated: 2026-11-17
---

# Registry entry for cognitive-kb

The initial FRINK registry entry for `cognitive-kb` was accepted through
[pull request 535](https://github.com/frink-okn/okn-registry/pull/535), merged
November 3, 2026. The block below is the proposed update for the current
document and topic concept model. Submit it as an update to
`docs/registry/kgs/cognitive-kb.md` through the
[registry contribution workflow](https://registry.okn.us/registry/contribute/).

The `sparql` and `tpf` fields are left out on purpose. The registry asks contributors to omit
them until the graph has been uploaded and converted, and RENCI adds them once the graph's own
endpoints exist. `funding` is left out because no grant funds the graph (see
[`graph_description.md`](graph_description.md)).

````markdown
---
template: overrides/kg.html
shortname: cognitive-kb
title: Cognitive Active Inference knowledge base
description: Concept pages on Active Inference and related fields, with typed and untyped relations and citation records, as SKOS/Dublin Core/schema.org RDF.
stats: https://registry.okn.us/kg-stats/cognitive-kb
homepage: https://github.com/ActiveInferenceInstitute/cognitive
frink-options:
  lakefs-repo: cognitive-kb
  documentation-path: cognitive-kb
contacts:
  - email: daniel@activeinference.institute
    github: docxology
    label: "Daniel Ari Friedman"
---

The knowledge base of the Active Inference Institute's `cognitive` repository as RDF. Every
page is a `schema:CreativeWork` document. Topic-bearing documents link through
`schema:about` and `dcterms:subject` to separate stable `/concept/<path>` SKOS
concepts in one concept scheme, grouped into domain `skos:Collection`s.
Conceptual relations have concept endpoints and explicit predicate meanings;
document references, citations, navigation, redirects and documentation relations
retain document endpoints. Ambiguous mixed-role assertions are preserved under
`https://purl.org/okn/frink/kg/cognitive-kb/relation/document/` without a SKOS
super-property. Body links use `dcterms:references`. Citation pages carry
authors, year and DOI (`schema:sameAs`). Explicit reviewed mappings use SKOS
predicates with source, rationale and review-date evidence. Unreviewed Wikidata
identifiers are `rdfs:seeAlso` discovery links. Tags point to concepts; citation
relations reuse BIBO. Provenance is `prov:wasDerivedFrom` the public
repository; repository-owned graph content is licensed under CC BY 4.0.
The graph serves public research and education by connecting explanatory concepts
and their literature across Active Inference, cognitive science, mathematics and
biology. SKOS correspondences apply to concept meanings and do not assert OWL
identity or clinical equivalence. No external grant funding is claimed.

A cross-graph query follows reviewed Wikidata mappings through the OKN
identifier-mappings graph to labeled OBO identifiers in Ubergraph. The query,
captured responses and replayable join evidence for the current corpus are in
[`docs/okn/evidence/2026-11-16-current-corpus`](https://github.com/ActiveInferenceInstitute/cognitive/tree/main/docs/okn/evidence/2026-11-16-current-corpus).
The join demonstrates identifier-based discovery while preserving the stated
mapping predicates. The reusable capture and submission methods are maintained
in `docxology/daf-okn`.

Entity IRIs are under `https://purl.org/okn/frink/kg/cognitive-kb/`; the named graph is
`https://purl.org/okn/frink/kg/cognitive-kb`. A prose description, vocabulary mapping, sample
and counts are in the source repository: <https://github.com/ActiveInferenceInstitute/cognitive/tree/main/docs/okn>.
The exporter, `code/scripts/export_rdf.py`, is in the same repository.
````

## Steps

1. Follow the [landing-zone handoff](landing_zone.md) to prepare, stage and
   verify the current graph against the [submission criteria](submission_criteria.md).
2. Submit the updated description and current query examples through the
   registry contribution workflow.
3. Verify processing with RENCI. RENCI adds the `sparql` and `tpf` fields when
   the endpoints are up. Keep the source revision and manifest with the lakeFS receipt.
