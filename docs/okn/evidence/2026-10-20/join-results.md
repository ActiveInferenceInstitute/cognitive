---
title: Captured OKN cross-graph join results
type: reference
status: stable
updated: 2026-10-20
---

# Captured OKN cross-graph join results

Observed at 2026-10-20T19:19:54.995793+00:00. Local cognitive-kb was joined with live OKN
identifier-mappings and Ubergraph responses, then replayed offline.

| Measurement | Value |
|---|---|
| Joined rows | 1 |
| Matched concept pages | 1 |
| Labeled external OBO terms | 1 |

The first edge is the local query's selected mapping from a concept
to Wikidata. The second is the mapping graph's `skos:exactMatch` to an
OBO identifier. Joining these edges does not establish identity between
the page and the ontology class. Labels are discovery signposts, not
clinical or scientific validation. Unmatched pages are omitted.

| Source page | Wikidata join key | External term |
|---|---|---|
| [Cortisol](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/cortisol.md) | [Q190875](http://www.wikidata.org/entity/Q190875) | [cortisol](http://purl.obolibrary.org/obo/CHEBI_17650) |
