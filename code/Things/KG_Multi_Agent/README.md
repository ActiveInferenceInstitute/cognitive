---
title: Knowledge Graph Multi-Agent Systems
type: concept
status: draft
created: 2025-01-01
updated: 2026-10-10
tags:
  - multi_agent
  - knowledge_graph
  - swarm_intelligence
semantic_relations:
  - type: related_to
    links:
      - "[[knowledge_base/cognitive/social_cognition|Social cognition]]"
      - "[[knowledge_base/systems/network_theory|Network theory]]"
      - "[[code/Things/Ant_Colony/README|Ant Colony]]"
---

# Knowledge Graph Multi-Agent Systems

Design sketch; no code in this repository. This folder documents ideas for
multi-agent systems that work on a knowledge graph. It holds no Python, no
`requirements.txt`, no tests, no configuration files, and no sample project
directories. Classes, commands, flags, and output files that earlier versions of
these pages described (`KnowledgeGraphMultiAgent`, `KnowledgeGraphProcessor`,
`infer_queries_batch.py`, `process_conversations.py`, `MKG_utils.py`, a
`test1/` project, a pytest suite, scaling and memory figures) do not exist
here.

## Contents

- [[code/Things/KG_Multi_Agent/Ant_Swarm|Ant Swarm]]: POMDP agents ("ants") traversing an Obsidian knowledge graph and leaving synthesized hypothesis nodes, read as stigmergy.
- [[code/Things/KG_Multi_Agent/KG_ActInf_Search|KG ActInf Search]]: an LLM supplies observations to a discrete POMDP whose hidden states are graph nodes; the chosen policy selects retrieval material.
- [[code/Things/KG_Multi_Agent/SWR_Swarm|SWR Swarm]]: hippocampal sharp-wave-ripple replay as a model for agent memory consolidation on the graph.
- [[code/Things/KG_Multi_Agent/MKG_Multi_Agent/README|MKG Multi-Agent]]: a concept for extracting research requests from agent conversation logs.

## Shared idea

Treat a linked markdown knowledge base as a graph of nodes with text, an
embedding, and neighbours. Agents start at nodes similar to a query, move along
links, collect evidence, and write a summary node back into the graph that other
agents can find. The coordination is indirect, through the shared graph, which
is why the notes use ant foraging and stigmergy as the reference model.

## What exists in the repository

- The vault itself is a linked markdown graph. `cognitive-kb-index` and
  `cognitive-verify-links` operate on it; see
  [[docs/api/api_reference|the API reference]].
- Working discrete and continuous agents live in `code/Things/Simple_POMDP` and
  `code/Things/Continuous_Generic`; the discrete dispatcher and generative model
  are in the `cognitive` package.

## Related

- [[knowledge_base/cognitive/swarm_intelligence|Swarm intelligence]]
- [[code/Things/BioFirm/README|BioFirm]]
- [[docs/research/ant_colony_active_inference|Ant colony active inference research]]
