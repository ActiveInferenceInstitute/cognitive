---
title: Vocabulary mapping for cognitive-kb
description: Which standard vocabulary term each part of a knowledge-base page becomes in the cognitive-kb graph, and the few terms the graph defines itself.
type: reference
status: stable
updated: 2026-10-20
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
| Topic-bearing page | Separate `…/concept/<path>` with `a skos:Concept`; `skos:inScheme <…/scheme>`. Document links with `schema:about` and `dcterms:subject` |
| `README`, `AGENTS`, `index`, `catalog` | `a schema:CollectionPage` |
| `title` | Document `dcterms:title`; topic `skos:prefLabel` |
| `aliases` | Document `schema:alternateName`; topic `skos:altLabel`. Topic aliases identical to its preferred title are omitted |
| `description` | `dcterms:description` |
| `created`, `updated` or `modified` | `dcterms:created`, `dcterms:modified` (`xsd:date`) |
| `tags` | `schema:keywords` to concept IRIs, never strings; reviewed terms are reused directly, other tags get named local concepts |
| `definition`, `scope_note` | Topic `skos:definition`, `skos:scopeNote`; document-only records retain these fields as `ckb:sourceDefinition`, `ckb:sourceScopeNote` |
| `type` | `ckb:pageType` |
| `status`, `complexity` | `ckb:status`, `ckb:complexity` |
| `processing_priority` | Document `ckb:processingPriority`, preserving the source editorial value |
| `authors`, `year`, `doi` (citation pages) | `schema:author`, `schema:datePublished`, `schema:sameAs` |
| `journal` (citation pages) | `ckb:journal` |
| `wikidata` (unreviewed discovery identifier) | `rdfs:seeAlso` to `http://www.wikidata.org/entity/Q…`; no SKOS mapping implied |
| `external_mappings` (source-reviewed records) | Mapping on the topic, with an `rdf:Statement` whose subject is that topic. Document-only roles cannot carry mappings |
| Explicit `license` URI | `dcterms:license` on the page; incompatible licenses fail `--check-submission` |
| Top-level folder | a `skos:Collection` that has topic concepts as `skos:member`; documents are not members |
| Address in the repository | `dcterms:source` |
| `redirect` | `dcterms:isReplacedBy` |

## Relations

| Written in the page | In the graph |
|---|---|
| `relates`, `related`, `relates_to`, `related_to` | `skos:related` between declared topics; otherwise a document-scoped source assertion |
| `part_of`, `component_of` | Conceptual `rel:part_of` / `rel:component_of` between topics; `dcterms:isPartOf` for document containment when an endpoint has no topic |
| `parent:` | Document `dcterms:isPartOf` |
| `organizes` | `dcterms:hasPart` |
| `[[wiki link]]` in the body | `dcterms:references` |
| `cited_by`, `cited by` | `bibo:citedBy` (`http://purl.org/ontology/bibo/citedBy`), cited document to citing document |
| `cites` | `bibo:cites`, citing document to cited document |
| `documents`, `documented_by`, `documented_in`, `defines`, `guides`, `maps`, `aggregates`, `explains`, `demonstrates`, `visualizes` | `rel:document/<name>` between documents, with an explicit written-material meaning |
| Reviewed conceptual names (`implements`, `extends`, `foundation`, `foundation_for`, `uses`, …) | `rel:<name>` between topics, a sub-property of `skos:semanticRelation` |
| Unknown names, unsupported explicit predicate IRIs, or conceptual assertions involving document-only roles | `rel:document/<normalized-name>` preserves the written assertion and exact source name; conceptual interpretation remains unresolved |

The [BIBO definitions](https://dcmi.github.io/bibo/) support the citation direction
already declared by citation-page frontmatter. This preserves the source assertion;
it does not independently verify the cited publication's bibliography. Generic
body links remain references. The [SIO definition](https://raw.githubusercontent.com/MaastrichtU-IDS/semanticscience/master/ontology/sio.owl)
requires an information entity conforming to a specification. Existing conceptual
`implements` edges often express broader relationships, so they retain their local
predicate. Likewise [CiTO extends](https://sparontologies.github.io/cito/current/cito.html#extends)
would assert an actual citation; conceptual extension alone is insufficient.
Known standard prefixes and absolute HTTP(S) predicate IRIs are parsed, then
checked against the routing policy. Only supported SKOS and document predicates
are emitted directly. Others retain their exact source spelling in
`ckb:sourceRelation` on a document-scoped predicate.

`skos:related` is symmetric in SKOS; the graph states each such edge in one direction only,
as the page wrote it. A consumer that wants the symmetric reading can add the inverse.

## Terms this graph defines

| Term | Meaning |
|---|---|
| `ckb:pageType` | The page's `type` value, for example `mathematical_concept`. |
| `ckb:status` | The page's editorial status. |
| `ckb:complexity` | The page's declared difficulty. |
| `ckb:processingPriority` | The page's source-declared editorial processing priority. |
| `ckb:journal` | The journal a citation page names. |
| `ckb:sourceDefinition`, `ckb:sourceScopeNote` | Original fields on a document without a declared topic. |
| `ckb:sourceRelation` | Exact source relation name defining a routed predicate. |
| `rel:<name>` | Reviewed conceptual relation with meaning/direction comment, Concept domain/range and SKOS semantic super-property. |
| `rel:document/<name>` | Source assertion with meaning/direction and CreativeWork domain/range, with no SKOS super-property. |

Complete source-to-target meanings are in `code/scripts/rdf_relations.py` and
emitted in RDF. `foundation_for`, `foundational_for` and `foundational_to` point
from a basis to the supported topic; `foundation`, `mathematical_basis`,
`foundational_theory`, `grounded_in` and `builds_on` point toward the basis.
`implements` connects a method/design concept to a specification/theory concept;
`implemented_by` runs the reverse direction. These preserve author assertions
without asserting executable software, verified causation or real-world class
membership. See [the migration contract](entity-model.md).

## Dataset description

The dataset is a `dcat:Dataset` and `void:Dataset` with `dcterms:title`, `dcterms:description`,
`dcterms:license`, `dcterms:creator`, `dcterms:publisher`, `dcterms:source`, `dcterms:modified`,
`dcat:landingPage` and `void:vocabulary` for each standard vocabulary above. Provenance uses
`prov:wasDerivedFrom`, `prov:wasGeneratedBy`, `prov:used` and `prov:wasAssociatedWith`.
`rdfs:seeAlso` on the dataset signposts the Wikidata, identifier-mappings and
Ubergraph registry entries. These links describe discovery paths, not additional
equivalence assertions or a claim that cognitive-kb is already deployed.

## Concept keywords

[Schema.org permits DefinedTerm, URL and Text keyword values](https://schema.org/keywords).
This export adopts Mahir's stricter concept-IRI profile. `model_fitting` directly
uses [STATO:0000218](https://www.ebi.ac.uk/ols4/ontologies/stato/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FSTATO_0000218),
the process of evaluating how a model represents a dataset; `computational_modeling`
uses [NCIT:C16461](https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/C16461?include=full),
the general computer-modeling technique. These are subject keywords, not types or
identity assertions about the publication. Additional reviewed tags use the exact
source-tag lookup in `scripts.rdf_keywords.KEYWORD_TERMS`, without network access.
Unmatched tags use `…/keyword/<encoded-tag>`
typed as `skos:Concept` and `schema:DefinedTerm`, with `skos:prefLabel`, exact source
spelling in `skos:notation`, and scheme membership. This preserves tags without
guessing an external equivalent. No external ontology classes are declared chemical
or process instances by this keyword reuse.

## Alignment to Wikidata and domain vocabularies

Unreviewed title/alias-based `wikidata` proposals are discovery links only.
[SKOS mapping relations](https://www.w3.org/TR/skos-reference/#mapping) require
semantic evidence even for `closeMatch`; confidence scores are not a substitute.
An explicit record has exactly these fields:

```yaml
external_mappings:
  - target: http://purl.obolibrary.org/obo/CHEBI_17650
    relation: exactMatch
    evidence: https://www.ebi.ac.uk/chebi/CHEBI:17650
    rationale: "Both source and target denote the cortisol chemical substance."
    reviewed_on: "2026-10-20"
```

Allowed relations are `exactMatch`, `closeMatch`, `broadMatch`, `narrowMatch` and
`relatedMatch`. A broader target uses `broadMatch`; an associative relationship
uses `relatedMatch`. Records with missing evidence, invalid IRIs/dates, unknown
fields or duplicate targets fail export. Navigation, citation, redirect and other
document-only pages cannot carry
these mappings. Each assertion has a stable, hashed `…/mapping/` statement with
`rdf:subject`, `rdf:predicate`, `rdf:object`, `dcterms:source`,
`dcterms:description` and `dcterms:date`. These document a source comparison by
the implementation agent; they do not claim independent human curation.

Existing page IRIs serve exclusively as document records. Separate stable
`…/concept/<path>` IRIs carry mappings and reified review subjects. Source,
rationale, target, relation and review date are preserved. Mapping to an external
OWL class uses its concept-level reading and does not assert `owl:sameAs`,
`owl:equivalentClass`, or membership of a particular chemical/process instance.
Cortisol now has an explicit chemical definition and scope note excluding the
HPA system, assays and drug products before its ChEBI/NCI exact matches are used.
The two free-energy-principle pages retain reviewed close matches because their
general self-organization scope is broader than Wikidata's neuroscience framing.
Active inference is only related to the free-energy-principle target.

See the generated [mapping review and priority queue](alignment-review.md) for
current correspondences and frequently referenced pages. No mapping is inferred
from a successful remote join, a similar label or a missing review record.

The [graph-wide quality method](graph-quality.md) specifies fatal SKOS integrity
checks, the local mapping-evidence and keyword profile, and nonfatal editorial
coverage. Use `--quality-report` for deterministic JSON and `--check-submission`
for the combined metadata and semantic gate. Neither interprets `exactMatch`
as identity substitution or treats hierarchy cycles alone as invalid.
