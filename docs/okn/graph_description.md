---
title: The cognitive-kb graph
description: A prose description of the cognitive-kb RDF graph, its nodes and edges, vocabularies, provenance, licence, funding and limits.
type: reference
status: stable
updated: 2026-10-12
---

# The cognitive-kb graph

`cognitive-kb` is the knowledge base of this repository as RDF: a graph of concept pages about
Active Inference and the fields it draws on, linked to one another. Figures are in
[`statistics.md`](statistics.md); a worked sample is in
[`cognitive-kb.sample.ttl`](cognitive-kb.sample.ttl).

## What the nodes are

Every Markdown page under `knowledge_base/` has a document node at its existing
`…/page/<path>` IRI, always a `schema:CreativeWork`. Topic-bearing documents link
to separate `…/concept/<path>` SKOS concepts with `schema:about` and
`dcterms:subject`. See the [entity model and migration](entity-model.md).

| Node | RDF type | What it is |
|---|---|---|
| Topic concept | `skos:Concept` in the `…/scheme` concept scheme | Subject of a topic-bearing document, such as Markov blanket, expected free energy or stigmergy. |
| Navigation page | `schema:CollectionPage` | A folder's `README`, `AGENTS`, `index` or `catalog` page. It lists other pages and defines nothing. |
| Redirect page | none beyond `schema:CreativeWork` | A retired name; `dcterms:isReplacedBy` points to the page that replaced it. |
| Citation record and other documentation | `schema:CreativeWork` | Published-work metadata, guides and editorial material; these do not become topic concepts. |

Pages are grouped into domains by top-level folder (`mathematics`, `biology`, `cognitive`,
`philosophy`, `systems`, `ontology`, `citations` and others). Each domain is a `skos:Collection`
that has its topic concepts as `skos:member`.

A document node carries:

- `dcterms:title` from the page title and `schema:alternateName` for aliases;
- `dcterms:description` (the page's one-line summary, where it has one), `dcterms:created` and
  `dcterms:modified` as `xsd:date`;
- `schema:keywords` concept IRIs for its tags, with reviewed external terms reused directly;
- `dcterms:source`, the page's address in the public repository;
- `ckb:pageType`, `ckb:status` and, where the page gives it, `ckb:complexity`.

Its topic carries `skos:prefLabel`, distinct aliases as `skos:altLabel`,
`skos:definition`, `skos:scopeNote`, scheme/domain membership and reviewed
external mappings. Page descriptions and source/license metadata stay on documents.

Citation pages, which record one published paper, also carry `schema:author`,
`schema:datePublished` (an `xsd:gYear`), `ckb:journal` and, when the page has a DOI,
`schema:sameAs` pointing at `https://doi.org/…`.

## What the edges are

Edges come from two places in a page; direction follows the source declaration.

1. **Typed edges** from the page's `semantic_relations` frontmatter, each a named relation
   such as `relates`, `implements`, `extends`, `foundation_for` or `cited_by`. The relations a
   standard vocabulary safely covers use it: `relates` becomes `skos:related`
   between topics, `organizes` becomes document `dcterms:hasPart`, and `cited_by`
   becomes document `bibo:citedBy`. Reviewed conceptual custom relations use
   `…/relation/` with Concept domain/range and a SKOS semantic super-property.
   Documentation and ambiguous mixed-role assertions use document-scoped
   predicates under `…/relation/document/` without SKOS entailment. Definitions,
   directions and endpoint scopes are emitted as RDF; unresolved conceptual
   interpretations are listed in the generated statistics.
2. **Untyped edges** from `[[wiki links]]` in the body, as `dcterms:references`. Links inside
   code and numeric matrix literals are ignored.

Relation names are the authors' own and are not normalised: `foundation_for` and
`foundational_for` are separate terms, and the direction is as written in the source page.
[`vocabulary.md`](vocabulary.md) lists the mapping.

Source-reviewed external correspondences use explicit SKOS predicates with
reified evidence, rationale and review date. Unreviewed Wikidata identifiers
are `rdfs:seeAlso` discovery links. The generated [review queue](alignment-review.md)
records current mappings and prioritizes heavily referenced concepts. Concept
alignment does not assert that a Markdown document is a molecule or that two
ontology classes are OWL-equivalent.

## Provenance

The graph is generated, not curated by hand. `dcat:Dataset` / `void:Dataset` `…/dataset` is
`prov:wasDerivedFrom` the public repository
<https://github.com/ActiveInferenceInstitute/cognitive> and `prov:wasGeneratedBy` the activity
`…/activity/export`, which `prov:used` that repository and was `prov:wasAssociatedWith` the
exporter source file. The dataset's `dcterms:modified` is the latest `updated` date among the
pages, so the output carries no timestamp and is identical on every run over the same pages.
Each page's content was written by Active Inference Institute contributors; the references on
each page name the published works it summarises.

## Public research and education use case

The graph supports finding concepts across Active Inference, cognitive science,
mathematics and biology, following their explanatory links, and locating the
literature they cite. Educators can build reading paths; researchers can join
concept pages to other datasets through shared identifiers and inspect the
original pages before interpreting a relation. These are public research and
education uses; the graph is not a source of clinical recommendations.

## Licence

The knowledge-base content, and so the graph, is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (`dcterms:license`
on the dataset and the concept scheme). The repository's code, including the exporter, is MIT.
Attribution and an indication of changes are required. The scope and suggested
attribution are in [the knowledge-base license](../../LICENSE-KNOWLEDGE-BASE.md).
References and links do not change the licenses of the external works they identify.

## Funding

No external grant funds this graph or its export. The pages are written and maintained by
volunteers of the [Active Inference Institute](https://activeinference.institute/), a
Delaware nonprofit. The registry entry therefore leaves its `funding` field out.

## Scope and limits

- The graph describes a body of explanatory writing, not measured data. An edge says that one
  page relates to or links another, not that the relation has been established in the
  literature.
- Concepts that pages mention but that have no page yet are not nodes; those links are counted
  in [`statistics.md`](statistics.md) and left out. So are links to files outside
  `knowledge_base/`, such as code and guides.
- Concept pages with an unreviewed `wikidata` field use discovery-only `rdfs:seeAlso`;
  explicit reviewed records use their stated SKOS mapping predicate. Citation
  pages with a DOI use `schema:sameAs`. Recorded comparisons do not constitute
  independent human review or clinical equivalence. See
  [interoperability evidence and review rules](interoperability.md).
- Self-links are dropped, and a page that appears under two names (an alias) is one node.
- [Semantic diagnostics](graph-quality.md) check labels, standard hierarchy and
  mapping conflicts, complete review statements, local keyword descriptions,
  identifiers, dates and citation endpoints across the export. Coverage metrics
  expose editorial gaps without making a missing definition or mapping fatal.
