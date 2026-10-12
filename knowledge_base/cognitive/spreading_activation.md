---
title: Spreading Activation
description: A model in which activation of one concept in a network spreads to related concepts, explaining semantic priming and memory retrieval, with a comparison to message passing in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - semantic_memory
  - priming
  - retrieval
semantic_relations:
  - type: relates
    links:
      - "[[semantic_memory]]"
      - "[[priming_effects]]"
      - "[[memory_organization]]"
      - "[[lexical_access]]"
      - "[[knowledge_representation]]"
wikidata: Q423068
---

# Spreading Activation

## Overview
Spreading activation is a processing assumption used in theories of memory and language. Knowledge is represented as a network of nodes, such as concepts or words, joined by links that reflect association or semantic relatedness. When one node is activated, activation propagates along the links to its neighbours, making them easier to access for a period of time. The idea explains why thinking of one item makes related items come to mind more quickly.

## Formulation and evidence
- **Semantic network theory.** Collins and Loftus (1975) developed a spreading-activation theory of semantic processing in which the strength of a link depends on how related two concepts are, and activation decreases with distance and over time.
- **Memory retrieval.** Anderson (1983) incorporated spreading activation into a theory of memory in which the activation of a memory element depends on its base level and on activation received from currently active elements, so that retrieval time and probability depend on the surrounding context.
- **Semantic priming.** A word such as "doctor" is recognised faster after a related word like "nurse" than after an unrelated one. Priming of this kind is the classic behavioural evidence for the mechanism, although priming can also arise from expectancy strategies and from overlap of features, so it does not uniquely identify spreading activation.
- **Applications.** The same principle appears in connectionist models, in cognitive architectures, and in information retrieval systems that rank documents by propagating relevance through a graph.

Critics note that the theory is flexible enough to accommodate many outcomes, and that parameters such as decay and spread are rarely fixed independently of the data.

## Relation to Active Inference
Active Inference does not use spreading activation as a primitive. The nearest correspondence is message passing on a graphical model, in which beliefs about one variable influence beliefs about connected variables through the links of the model (see [[message_passing]] and [[belief_propagation]]). Both frameworks describe local interactions propagating through a network, but they differ in kind: messages in belief propagation carry probabilistic information about states and obey rules fixed by the model, whereas spreading activation transmits an undifferentiated quantity with decay. The resemblance is an analogy, not an equivalence.

## Related concepts
- [[semantic_memory]] — the network of concepts across which activation spreads
- [[priming_effects]] — the principal empirical phenomenon explained
- [[memory_organization]] — structure of stored knowledge
- [[lexical_access]] — word retrieval, a main domain of application
- [[knowledge_representation]] — formats that the network assumption instantiates
- [[message_passing]] — the Active Inference analogue of propagation through a network

## References
- Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. Psychological Review, 82, 407-428.
- Anderson, J. R. (1983). A spreading activation theory of memory. Journal of Verbal Learning and Verbal Behavior, 22, 261-295.
