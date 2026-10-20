---
title: OKN submission criteria and evidence
type: reference
status: stable
updated: 2026-10-20
---

# OKN submission criteria and evidence

This page records the requirements in the email from Jim Balhoff (RENCI) supplied
by Daniel on 2026-10-13. Jim described an early criteria document that was not
yet published. This is a repository interpretation of that supplied text;
RENCI decides whether to accept the graph. The
[registry contribution guide](https://registry.okn.us/registry/contribute/) and
[upload setup guide](https://registry.okn.us/help/setup/) describe the public workflow.

| Supplied criterion | Repository evidence | Review boundary |
|---|---|---|
| Clear public research, education or public-good innovation use case | [Graph description](graph_description.md): concept discovery, reading paths and literature links | RENCI assesses thematic fit |
| CC-BY or CC0 graph license | [CC BY 4.0 notice](../../LICENSE-KNOWLEDGE-BASE.md); `dcterms:license` on dataset and concept scheme | Maintainer approved the change and reported no known third-party content exceptions; checks cannot establish ownership |
| Recommended domain standards and identifiers | SKOS, Dublin Core, schema.org, PROV-O, DCAT, VoID; existing Wikidata and DOI links | Apply domain identifiers only when a page denotes the appropriate entity |
| Good-faith evidence of mapping and meshing with OKN graphs | [Interoperability](interoperability.md), [captured join results](evidence/2026-10-20-graph-methods/join-results.md), runnable cross-graph query | Local export joined with live identifier-mappings and Ubergraph responses; hosting cognitive-kb and server-side execution remain separate steps |
| Federal-program alignment is encouraged, not required | Existing description reports volunteer maintenance without external grant support | Do not claim an NSF or NIH award for this graph |

## Before upload

1. Review licensing scope and any new contributor or third-party notices. An
   explicit `license` frontmatter field is retained in RDF; an incompatible
   license fails the submission check. This check reads exported metadata and
   cannot detect undeclared restrictions in prose or external sources.
2. Regenerate the sample, statistics and alignment review with `uv run --extra dev
   python -m scripts.export_rdf --write-docs`. Run the repository quality gates and `uv run --extra dev python -m
   scripts.export_rdf --check-submission`. Examine unresolved link counts; these
   links are omitted from the graph and must not be described as exported edges.
3. Export the graph and retain `manifest.json`, the source commit, and the
   review of any local changes. Hashes identify content; a commit identifies history.
4. Review identifier correspondence, execute the example queries, and replay the
   captured join using daf-okn. Preserve the source export, bounded queries,
   remote responses and hashes. Recheck the federation query after deployment.
5. Update the registry description and obtain the landing-zone access specified
   by the public workflow. Use [the landing-zone handoff](landing_zone.md) and
   record the actual upload and processing receipts.

Mahir's subsequent semantic review is implemented in the [vocabulary contract](vocabulary.md):
concept-valued keywords, standard citation predicates, discovery-only unreviewed
links and explicit evidence for each SKOS correspondence. Submission checks reject
literal keywords and mapping assertions without complete review records. A passing
check verifies that evidence is recorded; it does not verify the scientific judgment.

The combined check now includes the [graph-wide semantic quality profile](graph-quality.md).
Run `uv run --extra dev python -m scripts.export_rdf --quality-report` to inspect
its deterministic JSON separately. Fatal findings include SKOS label and relation
disjointness, malformed identifiers or typed dates, incomplete or orphan mapping
review statements, incomplete local keyword descriptions and invalid citation
endpoints. Missing editorial fields and SKOS-permitted hierarchy cycles remain
nonfatal; inspect coverage and warnings before upload. `--check-submission` embeds
the same semantic report under `quality` and returns status one for any fatal
metadata or semantic finding.

The repository's automated check is a necessary technical check, not a full
submission certification. Generated exports, registry acceptance, upload,
processing and federation availability are separate events.
