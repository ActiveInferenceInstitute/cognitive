---
title: Captured OKN cross-graph join results
type: reference
status: stable
updated: 2026-10-23
---

# Captured OKN cross-graph join results

Observed at 2026-10-23T17:31:45.038105+00:00. Local cognitive-kb was joined with live OKN
identifier-mappings and Ubergraph responses, then replayed offline.

| Measurement | Value |
|---|---|
| Joined rows | 2 |
| Matched documents | 2 |
| Matched concepts | 2 |
| Labeled external OBO terms | 2 |

Each document describes a separate concept through `schema:about` and
`dcterms:subject`. The concept's reviewed Wikidata mapping is joined to
the remote mapping graph's `skos:exactMatch` and an OBO label.
Document identifiers, concept identifiers and Markdown sources remain
distinct. Joining these edges adds no identity or class-instance assertion.
Labels are discovery signposts, not clinical or scientific validation.
Unmatched documents are omitted.

| Document | Concept | Wikidata join key | External term | Source |
|---|---|---|---|---|
| [Document](https://purl.org/okn/frink/kg/cognitive-kb/page/biology/cortisol) | [Cortisol](https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/cortisol) | [Q190875](http://www.wikidata.org/entity/Q190875) | [cortisol](http://purl.obolibrary.org/obo/CHEBI_17650) | [Markdown](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/cortisol.md) |
| [Document](https://purl.org/okn/frink/kg/cognitive-kb/page/biology/symbiosis) | [Symbiosis](https://purl.org/okn/frink/kg/cognitive-kb/concept/biology/symbiosis) | [Q121610](http://www.wikidata.org/entity/Q121610) | [biological process involved in symbiotic interaction](http://purl.obolibrary.org/obo/GO_0044403) | [Markdown](https://github.com/ActiveInferenceInstitute/cognitive/blob/main/knowledge_base/biology/symbiosis.md) |
