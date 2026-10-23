---
title: Graph-wide semantic quality method
type: reference
status: stable
updated: 2026-10-20
---

# Graph-wide semantic quality method

Run `uv run --extra dev python -m scripts.export_rdf --quality-report` from the
repository root. It prints deterministic JSON without writing an export or changing
the vault. Exit status is zero when semantic errors are absent and one when any
semantic error exists. Warnings do not change the status. Invalid mapping records
raise parsing errors with a nonzero exit status rather than a passing report.

`scripts.rdf_quality.semantic_quality(graph)` is the importable API. The report
has `schema_version`, `profile`, `machine_checks_passed`, `errors`, `warnings`,
`finding_counts`, `coverage` and `work`. Findings carry a stable `code`, severity,
message and identifying terms. Terms are serialized RDF terms, except decoded
label text and language in label conflicts. Arrays and counts have a fixed order;
there is no timestamp, duration or network-dependent result. The separated entity
model uses profile `cognitive-kb-semantic-2` and schema version `2`.

`scripts.export_rdf.submission_diagnostics(graph)` adds the existing license,
nonempty-page, page-reference, interoperability-presence and provenance criteria.
Its `quality` field contains the semantic report. `submission_errors(graph)`
preserves the list-of-strings interface. `--check-submission` returns the combined
diagnostics and blocks on either metadata or semantic errors. The semantic-only
command can pass a graph without mappings; the submission command still requires
a mapping on a document's topic as evidence of an interoperability review.

## Standards and fatal errors

The checker implements a bounded set of entailments from the
[W3C SKOS Reference](https://www.w3.org/TR/skos-reference/). It does not perform
general OWL reasoning or fetch external ontology definitions.

- [S13 and S14](https://www.w3.org/TR/skos-reference/#labels): preferred,
  alternative and hidden label properties are pairwise disjoint. Preferred
  labels have at most one value per language. Checks compare decoded RDF text
  and case-insensitive language tags, permitting distinct languages. Plain
  strings and `xsd:string` have the same label text interpretation.
- [S27](https://www.w3.org/TR/skos-reference/#semantic-relations): `related` is
  disjoint with transitive hierarchy. Hierarchy includes `broader`, `narrower`,
  their transitive properties, `broadMatch` and `narrowMatch`. The checker follows
  inverse directions and hierarchy paths; `relatedMatch` participates as a
  sub-property of `related`.
- [S46](https://www.w3.org/TR/skos-reference/#mapping): `exactMatch` conflicts
  with `broadMatch`, its inverse `narrowMatch`, and `relatedMatch` on the same
  endpoints. The checker includes symmetric and transitive `exactMatch` closure.
  It does not close `closeMatch` transitively or treat exact matches as resource
  identity. It does not infer arbitrary property substitution through mappings.
- SKOS semantic relation properties require resource endpoints. This export's
  local profile requires absolute HTTP(S) IRIs for those resources.

Cycles alone are warnings: SKOS explicitly permits them in
[hierarchical relations](https://www.w3.org/TR/skos-reference/#L2484).
`exactMatch` plus `closeMatch` is allowed. A missing preferred label, definition,
scope note, description, or mapping is not by itself a SKOS integrity violation.

## Local submission profile

These additional constraints are repository policy, not new ontology axioms:

- Every local `/page/` resource is a CreativeWork document, never a Concept or
  SKOS mapping endpoint. Each `/concept/` topic is a Concept in the local scheme
  linked from its matching document through both `schema:about` and
  `dcterms:subject`. Navigation, citations, redirects and other document-only
  types cannot acquire topics from typed links. Document metadata and citation
  predicates cannot be placed on topics; SKOS concept metadata cannot be placed
  on documents. Domain memberships require concepts.
- Local conceptual predicates require an explicit reviewed meaning, source name,
  Concept domain/range and SKOS semantic super-property. Document-scoped source
  predicates require CreativeWork domain/range and no SKOS super-property,
  including indirectly inherited super-properties. Both endpoint types are
  checked directly. Standard SKOS relation/mapping endpoints cannot be pages even
  if their explicit Concept types were removed. Unknown source relation names
  are preserved as documentary assertions rather than assigned a conceptual meaning.

- Every asserted SKOS mapping needs an asserted `rdf:Statement` with exactly one
  `rdf:subject`, `rdf:predicate`, `rdf:object`, HTTP(S) evidence source, nonempty
  `@en` rationale, and valid review date. A local resource used as a mapping
  source must have its concept declaration. Every statement is checked, even
  when another valid statement covers the same mapping. Detached metadata,
  malformed statements and statements whose mapping triple is absent fail.
  This profile reserves RDF reification in the export for mapping reviews.
- Local keyword resources need `skos:Concept`, `schema:DefinedTerm`, preferred
  label, nonempty notation, and the local concept scheme. External keywords
  need valid IRIs; the checker does not copy or invent their declarations.
- The profile uses HTTP(S) IRIs without credentials or control characters, valid
  percent escapes and a nonempty authority. Explicit ports must be integers
  from 1 through 65535. Empty port suffixes are rejected by local policy.
  These checks build on [RFC 3986](https://www.rfc-editor.org/rfc/rfc3986)
  and do not test availability, redirects, ownership or identifier meaning.
- Typed calendar dates use actual Gregorian `YYYY-MM-DD` dates and publication
  years use four digits. This is a narrower local serialization profile than
  [XML Schema date and gYear](https://www.w3.org/TR/xmlschema11-2/), which also
  allow timezone and other year forms. Impossible typed dates fail. Untyped
  page dates remain visible as warnings rather than being discarded.
- Citation predicates use [BIBO](https://dcmi.github.io/bibo/): `cites` runs from
  citing document to cited document; `citedBy` runs in the opposite direction.
  Both require HTTP(S) endpoints and declared local pages where applicable.
  The report normalizes both predicates to distinct citing/cited pairs for
  coverage. It cannot determine whether a bibliography supports an assertion
  or identify an accidentally reversed source statement from labels alone.
  Custom `rel:cites` and `rel:cited_by` fail the profile.

The exporter omits a concept alias identical to its preferred title rather than
emitting an S13 violation. Document aliases retain their source spelling through
`schema:alternateName`. See the [entity model](entity-model.md) and
[vocabulary contract](vocabulary.md) for field routing and migration.

## Editorial coverage and cost

Coverage records local concept and page counts, preferred labels, definitions,
descriptions and scope notes, keyword descriptions, mapping assertions and
evidenced assertions, citation records and identifiers, normalized citation
pairs and hierarchy cycle components. Page-topic concept coverage is reported
separately from keyword concepts so a large keyword vocabulary does not inflate
topic definition coverage. `page_concepts` retains its report key and now counts
separate `/concept/` resources. `page_concepts_with_description` counts their
documents' descriptions, while `concepts_with_description` counts literal concept
descriptions (normally absent). `pages_with_reviewed_mapping` counts documents
whose topics are mapped. `document_topic_links` and
`document_relation_assertions` make the routes observable. Missing labels and language-neutral labels have
locatable warnings. Low definition or mapping coverage is an editorial review
signal, not a requirement to invent definitions or correspondences.

`GraphIndex` builds subject/predicate and predicate indexes in one traversal.
Statement fields use keyed lookups, so expanding mapping review does not rescan
the full graph for every mapping. `work.triples_indexed` and
`work.mapping_statement_lookups` expose deterministic work counts. Exact-match
components and hierarchy cycles use iterative component discovery. Related vs
hierarchy checks search only hierarchy adjacency; a dense hierarchy may still
require substantial reachability work. The checker does not materialize all
inferred hierarchy pairs or exact-match triples.

`code/tests/test_rdf_quality.py` exercises real RDF graphs, actual temporary
vaults, inverse and closure conflicts, every evidence field, orphan statements,
malformed identifiers/dates, language-sensitive labels, permitted cycles, and
a deterministic graph larger than the current vault with many reviewed mappings.
The scale test checks counts and a corrupted late review; it has no timing
threshold. `code/tests/test_export_rdf.py` covers export and submission integration.
Passing these checks records structural consistency and review metadata; it does
not certify scientific equivalence, independent human review or RENCI acceptance.

## Exporter runtime source inventory

The export manifest hashes the following repository runtime sources, in addition
to all root Markdown, all Markdown under `knowledge_base/`, `docs/` and `code/`,
and existing `pyproject.toml`, `uv.lock` and `.python-version` files. This is a
conservative superset of Markdown inputs to link resolution, including generated
documentation and the knowledge-base license. `RUNTIME_SOURCE_NAMES` in
`scripts.export_rdf` is the canonical module inventory:

- `code/scripts/__init__.py`
- `code/scripts/export_rdf.py`
- `code/scripts/rdf_alignment.py`
- `code/scripts/rdf_quality.py`
- `code/scripts/rdf_keywords.py`
- `code/scripts/rdf_relations.py`
- `code/scripts/kb_index.py`
- `code/scripts/verify_links.py`
- `code/scripts/markdown_fences.py`

The keyword registry and relation policy affect subject reuse and entity routing.
The last three implement frontmatter parsing, link resolution and Markdown code
exclusion; they affect graph construction even though they predate diagnostics.
The exporter adds no runtime dependency. A normal export hashes the source set
before graph construction and rechecks its inventory and bytes before writing
artifacts. A changed inventory or hash aborts the export. Markdown suffix matching
is case-insensitive on every platform, consistent with link resolution. Source
files must be regular files without symlinks, including in read-only knowledge-base
checks. Read-only diagnostics do not write a manifest.

The manifest also records `resolution_files`: sorted unique names of regular
root files and every regular file under `knowledge_base/`, `docs/` and `code/`.
Adding even an extensionless file can shadow a wiki target without changing any
Markdown bytes. This name inventory is checked before and after construction;
symlinks, special files and traversal errors fail. Cache/generated directories
within these scoped trees are included in this conservative inventory.

The hashes bind Markdown/runtime/configuration contents; other scoped files are
name-bound. They do not snapshot an executing interpreter, installed dependency
binaries, or other checkout trees. The bundle's source commit and dirty-state
receipt remain separate evidence. This is bounded custody, not a hermetic build
claim or an arbitrary filesystem snapshot.
