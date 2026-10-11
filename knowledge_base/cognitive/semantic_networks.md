---
title: Semantic Networks
description: Graph models of conceptual knowledge in which concepts are nodes and relations are links, covering hierarchical and spreading-activation models, their evidence and links to graph-based knowledge representation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - semantic_memory
  - knowledge_representation
  - networks
semantic_relations:
  - type: relates
    links:
      - "[[semantic_memory]]"
      - "[[spreading_activation]]"
      - "[[knowledge_representation]]"
      - "[[knowledge_graphs]]"
      - "[[hierarchical_processing]]"
---

# Semantic Networks

## Overview
A semantic network represents conceptual knowledge as a graph in which nodes stand for concepts and labelled links stand for relations such as "is a", "has" or "is part of". The idea originated in artificial intelligence and was adopted in cognitive psychology as a model of how knowledge in semantic memory is organised and retrieved. Related knowledge-engineering formalisms appear in [[knowledge_graphs]].

## Hierarchical network model
Collins and Quillian (1969) proposed that concepts are stored in a taxonomic hierarchy, with properties stored at the highest level where they apply. "Canary" links to "bird", which links to "animal". Under the principle of cognitive economy, a property such as "has wings" is stored with "bird" and not repeated for each species, and verifying a statement requires travelling up the hierarchy. The model predicted that verification time increases with the number of links traversed, and early reaction-time data appeared to agree. Later studies found effects that the model could not accommodate, such as typicality, since people confirm "a robin is a bird" faster than "a penguin is a bird", and the model's strict hierarchy was relaxed.

## Spreading activation
Collins and Loftus (1975) described a more flexible network in which links vary in strength and are organised by semantic similarity. Activating a concept spreads activation to its neighbours, which explains priming: reading "doctor" speeds the recognition of "nurse". Semantic priming is a robust finding and a standard tool for probing lexical and conceptual organisation (see [[spreading_activation]]). Network ideas continue in connectionist models, where concepts are distributed patterns and not single nodes, and in analyses of large word-association networks using graph measures such as clustering and path length.

## Relation to Active Inference
The connection is loose. Both use graph structure to organise knowledge, and hierarchical generative models can be seen as encoding taxonomic and part-whole relations. Active Inference models, however, represent knowledge as probability distributions over states and observations and not as symbolic links, and spreading activation is a process model without a derivation from free energy minimisation.

## Related concepts
- [[semantic_memory]] — the memory system that networks model
- [[spreading_activation]] — the retrieval process over network links
- [[knowledge_representation]] — the wider question of how knowledge is encoded
- [[knowledge_graphs]] — a related formal and engineering tradition
- [[hierarchical_processing]] — organisation across levels

## References
- Collins, A. M., & Quillian, M. R. (1969). Retrieval time from semantic memory. Journal of Verbal Learning and Verbal Behavior, 8, 240-247.
- Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. Psychological Review, 82, 407-428.
