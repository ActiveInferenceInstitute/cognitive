---
title: Vocabulary mapping for cognitive-kb
description: Which standard vocabulary term each part of a knowledge-base page becomes in the cognitive-kb graph, and the few terms the graph defines itself.
type: reference
status: stable
updated: 2026-10-12
---

# Vocabulary mapping

The graph reuses standard vocabularies wherever one fits and defines a term of its own only
where none does. Prefixes: `skos` = `http://www.w3.org/2004/02/skos/core#`, `dcterms` =
`http://purl.org/dc/terms/`, `schema` = `https://schema.org/`, `prov` =
`http://www.w3.org/ns/prov#`, `dcat` = `http://www.w3.org/ns/dcat#`, `void` =
`http://rdfs.org/ns/void#`, `ckb` = `https://purl.org/okn/frink/kg/cognitive-kb/vocab/`,
`rel` = `https://purl.org/okn/frink/kg/cognitive-kb/relation/`.

## Page fields

| In the page | In the graph |
|---|---|
| Page (any) | `a schema:CreativeWork` |
| Concept page | `a skos:Concept`; `skos:inScheme <…/scheme>` |
| `README`, `AGENTS`, `index`, `catalog` | `a schema:CollectionPage` |
| `title` | `dcterms:title`; for a concept also `skos:prefLabel` |
| `aliases` | `skos:altLabel` |
| `description` | `dcterms:description` |
| `created`, `updated` or `modified` | `dcterms:created`, `dcterms:modified` (`xsd:date`) |
| `tags` | `schema:keywords` |
| `type` | `ckb:pageType` |
| `status`, `complexity` | `ckb:status`, `ckb:complexity` |
| `authors`, `year`, `doi` (citation pages) | `schema:author`, `schema:datePublished`, `schema:sameAs` |
| `journal` (citation pages) | `ckb:journal` |
| `wikidata` (a `Q` identifier of the Wikidata item with the same meaning; see [Alignment to Wikidata](#alignment-to-wikidata)) | `skos:closeMatch` to `http://www.wikidata.org/entity/Q…` |
| Explicit `license` URI | `dcterms:license` on the page; incompatible licenses fail `--check-submission` |
| Top-level folder | a `skos:Collection` that has the page as `skos:member` |
| Address in the repository | `dcterms:source` |
| `redirect` | `dcterms:isReplacedBy` |

## Relations

| Written in the page | In the graph |
|---|---|
| `relates`, `related`, `relates_to`, `related_to` | `skos:related` |
| `part_of`, `component_of`, `parent:` | `dcterms:isPartOf` |
| `organizes` | `dcterms:hasPart` |
| `[[wiki link]]` in the body | `dcterms:references` |
| any other name (`implements`, `extends`, `foundation`, `foundation_for`, `cited_by`, `uses`, …) | `rel:<name>`, a sub-property of `skos:semanticRelation` |

`skos:related` is symmetric in SKOS; the graph states each such edge in one direction only,
as the page wrote it. A consumer that wants the symmetric reading can add the inverse.

## Terms this graph defines

| Term | Meaning |
|---|---|
| `ckb:pageType` | The page's `type` value, for example `mathematical_concept`. |
| `ckb:status` | The page's editorial status. |
| `ckb:complexity` | The page's declared difficulty. |
| `ckb:journal` | The journal a citation page names. |
| `rel:<name>` | A relation name from `semantic_relations` that no standard vocabulary covers. Each is declared in the graph with `rdfs:label` and `rdfs:subPropertyOf skos:semanticRelation`. |

## Dataset description

The dataset is a `dcat:Dataset` and `void:Dataset` with `dcterms:title`, `dcterms:description`,
`dcterms:license`, `dcterms:creator`, `dcterms:publisher`, `dcterms:source`, `dcterms:modified`,
`dcat:landingPage` and `void:vocabulary` for each standard vocabulary above. Provenance uses
`prov:wasDerivedFrom`, `prov:wasGeneratedBy`, `prov:used` and `prov:wasAssociatedWith`.
`rdfs:seeAlso` on the dataset signposts the Wikidata, identifier-mappings and
Ubergraph registry entries. These links describe discovery paths, not additional
equivalence assertions or a claim that cognitive-kb is already deployed.

## Alignment to Wikidata

Concept pages carry an optional `wikidata: Q…` key that the export writes as `skos:closeMatch`.
The identifiers were proposed by searching Wikidata for each page title and alias, keeping only
items whose label matches, and then screened by a language model that compared the item's
description with the page's. A second, independent pass re-judged every match and dropped the
ones it doubted. A person has not checked them one by one, so they are published as
`skos:closeMatch`, not `skos:exactMatch`: the items agree in meaning with the page, but a wrong
one may remain. Pages without a confident match have no key, and a missing link is preferred to a
wrong one. To report or fix a match, edit the page's `wikidata` value.
