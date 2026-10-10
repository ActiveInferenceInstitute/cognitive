---
title: MKG Multi-Agent
type: concept
status: draft
updated: 2026-10-10
semantic_relations:
  - type: related_to
    links:
      - "[[code/Things/KG_Multi_Agent/README|KG Multi-Agent overview]]"
---

# MKG Multi-Agent

Design sketch; no code in this repository. This folder contains only this
page. Earlier versions described scripts (`infer_queries_batch.py`,
`process_conversations.py`, `MKG_utils.py`), a `test1/` project tree, a
requirements list, command-line flags, a cultural-marker configuration, and
generated output files. None of those exist here.

## Concept

A pipeline that turns logs of AI-agent conversations into structured knowledge
notes:

1. Split long conversations into chunks that respect speaker boundaries.
2. Ask an LLM to extract research requests: a hypothesis, a rationale, the
   expected impact, the agents involved, and tags.
3. Correct misspelled agent names by fuzzy matching against the speakers seen in
   the conversation.
4. Write each research request as an Obsidian-compatible markdown note with
   frontmatter and links, so the extracted knowledge joins the vault's graph.
5. Optionally build interaction-network and participation statistics over the
   agents.

## Repository support

The vault conventions that the output notes would need (frontmatter, quoted
wiki links) are checked by `cognitive-validate-docs` and
`cognitive-normalize-frontmatter`; see [[docs/api/api_reference|the API reference]].
The extraction pipeline itself would have to be written.
