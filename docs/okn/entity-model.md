---
title: Document and topic concept model
type: reference
status: stable
updated: 2026-10-23
---

# Document and topic concept model

Every existing `…/page/<path>` IRI identifies a Markdown document, typed
`schema:CreativeWork`. A declared topic has a separate stable
`…/concept/<path>` IRI, typed `skos:Concept`. Its document links to it through
both `schema:about` and `dcterms:subject`. For example, the Markov blanket
document retains its title, created date, source, tags and editorial status;
the Markov blanket topic carries SKOS labels and conceptual `foundation_for`
and `implements` assertions. Neither document nor topic is asserted to be a
real-world Markov blanket instance.

## Roles and field ownership

`TOPIC_PAGE_TYPES` in `code/scripts/rdf_relations.py` is the explicit list of
topic-bearing source types. It includes mathematical, biological and cognitive
concepts, domain-topic articles, systems and ontologies. These are vocabulary
topics even when they describe a class, process or physical substance; export
does not instantiate that class. A type-less record with reviewed mappings is
also an explicitly mapped topic. A missing or unrecognized type alone never
creates a topic.

README, AGENTS, catalog and index navigation, citation records and citation
redirects, guides, implementation instructions, resources, examples and other
document-only types remain documents. Citation-directory records cannot acquire
topics from an incidental source type. `citation_redirect` and `redirect` retain
document-to-document replacement links. A proposed topic for a guide requires
source review; typed outgoing links alone are insufficient evidence.

Document titles, descriptions, aliases (`schema:alternateName`), dates,
authors, year, journal, identifiers, repository source, license, editorial fields
and keywords remain on documents. Topic labels, aliases (`skos:altLabel`),
definition, scope note, concept-scheme membership, domain membership and reviewed
SKOS mappings belong to topics. Definitions/scope notes on document-only records
are preserved as `ckb:sourceDefinition` / `ckb:sourceScopeNote`; they do not
silently change the role. Unreviewed Wikidata discovery links stay `rdfs:seeAlso`
on documents.

Tags are document subject metadata. The exact source spelling is looked up in
`scripts.rdf_keywords`; reviewed external terms are reused directly. Otherwise
the tag gets a local `…/keyword/` Concept/DefinedTerm with label, source notation
and scheme. Neither route asserts document identity or class membership. Domain
collections contain topic concepts, never documents or citation records.

## Relation routing and residual ambiguity

All directions follow the source declaration. `relates*` becomes `skos:related`
only between topics. Reviewed custom conceptual relations have explicit meanings
in `CONCEPT_RELATIONS`, emitted comments, Concept domain/range and the
`skos:semanticRelation` super-property. `foundation_for` points from foundation
to supported topic; `foundation` points toward the foundation. `implements`
points from method/design topic to specification/theory topic. These are
source-authored conceptual assertions, not execution or causal verification.

Body references, BIBO citations, redirects, navigation containment and
documentation relations remain document-valued. Part/component assertions between
topics use conceptual predicates; those involving document-only endpoints retain
document containment. Unknown relation names, unsupported explicit predicate IRIs
and conceptual labels on mixed-role endpoints are preserved under
`…/relation/document/<normalized-name>`. The predicate records its exact source
name, direction, meaning and CreativeWork scope without a SKOS super-property.
For ambiguous assertions its meaning is limited to the source document listing
the target under that label. This retains the source evidence without deciding
whether the author meant documentation, implementation, a topic or a publication.
No inverse or stronger standard relation is invented.

`Report.document_relation_routes` and the generated statistics table expose
individual source/target paths, relation names and reasons for review. A source
assertion to an unresolved or outside-knowledge-base target retains the existing
explicit omission counts. Consumers must inspect these counts before claiming
export completeness.

## Migration and query changes

Page IRIs and source Markdown are preserved. Reviewed mapping relations, targets,
evidence URLs, rationale text and dates migrate intact to their local topics.
Multiple reviewed targets remain separate assertions; export makes no arbitrary
choice of one external representative. Reified mapping subjects and their hashed
statement IRIs change because the subject has changed. No `owl:sameAs`,
`owl:equivalentClass` or identity substitution is added.

Queries now join `?page schema:about ?concept` before reading document source
metadata. Mapping and review queries return both variables; citation and keyword
queries retain document subjects. The runnable `queries/document_topics.rq` and
`queries/relation_scopes.rq` expose the two identities and predicate meanings.
The cross-graph query still follows reviewed Wikidata concept mappings. Earlier
captures retain their original bytes and model; fresh captures and replay are
required for the migrated export.

The [semantic profile](graph-quality.md) rejects dual document/concept typing,
page mapping endpoints, wrong document-topic links, concept-held document
metadata/citations and incorrect relation scopes. Existing SKOS integrity and
mapping-evidence checks remain active. This structural gate does not certify
scientific correctness, independent review or registry acceptance.
