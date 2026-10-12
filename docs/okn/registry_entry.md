---
title: OKN registry entry for cognitive-kb
description: The registry entry file to submit for cognitive-kb, and the steps from export to a listed graph.
type: guide
status: stable
updated: 2026-10-12
---

# Registry entry for cognitive-kb

The FRINK registry takes a graph as a pull request that adds one Markdown file to
`docs/registry/kgs/` of [`frink-okn/okn-registry`](https://github.com/frink-okn/okn-registry),
following its [contribution guide](https://registry.okn.us/registry/contribute/). Save the
block below as `docs/registry/kgs/cognitive-kb.md`.

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
page is a `schema:CreativeWork`; articles on one idea are also `skos:Concept`s in one concept
scheme, grouped into `skos:Collection`s by domain (mathematics, biology, cognitive science,
philosophy, systems, ontology, citations). Edges are typed relations from each page's
`semantic_relations` (`skos:related`, `dcterms:isPartOf`, `dcterms:hasPart` or a term under
`https://purl.org/okn/frink/kg/cognitive-kb/relation/` that is a sub-property of
`skos:semanticRelation`) and untyped body links (`dcterms:references`). Citation pages carry
authors, year and DOI (`schema:sameAs`). Provenance is `prov:wasDerivedFrom` the public
repository; the licence is CC BY-NC-SA 4.0.

Entity IRIs are under `https://purl.org/okn/frink/kg/cognitive-kb/`; the named graph is
`https://purl.org/okn/frink/kg/cognitive-kb`. A prose description, vocabulary mapping, sample
and counts are in the source repository: <https://github.com/ActiveInferenceInstitute/cognitive/tree/main/docs/okn>.
The exporter, `code/scripts/export_rdf.py`, is in the same repository.
````

## Steps

1. `python -m scripts.export_rdf` writes `output/okn/cognitive-kb.nq`.
2. Open the registry pull request with the entry above.
3. When it is merged, upload `cognitive-kb.nq` to the `cognitive-kb` lakeFS repository as the
   registry's [setup guide](https://registry.okn.us/help/setup/) describes.
4. Nothing else is needed from the contributor; RENCI adds the `sparql` and `tpf` fields when
   the endpoints are up.
