---
title: Analogical Reasoning
description: Reasoning that transfers knowledge from a familiar source to a new target by mapping relational structure, covering structure-mapping theory, constraint models, transfer in problem solving, and links to model-based inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - reasoning
  - analogy
  - transfer
semantic_relations:
  - type: relates
    links:
      - "[[inductive_reasoning]]"
      - "[[category_learning]]"
      - "[[reasoning_problem_solving]]"
      - "[[working_memory]]"
      - "[[abductive_reasoning]]"
aliases:
  - analogy
---

# Analogical Reasoning

## Overview
Analogical reasoning draws inferences about a target situation from its similarity to a better known source. An analogy between the solar system and an atom, or between blood circulation and a hydraulic circuit, lets properties of one domain be projected onto the other. Analogy underlies explanation, scientific discovery, legal argument and teaching, and it is a core component of human fluid intelligence.

## Structure mapping
Gentner's structure-mapping theory (1983) holds that an analogy is a mapping of relations between objects, not of the objects' own attributes. The theory proposes that:
- Relations that are part of a connected, higher-order system, for example causal relations, are preferred over isolated matches (the systematicity principle).
- Mappings are one-to-one, so each source element corresponds to at most one target element.
- Inferences arise by carrying over unmatched relations from the source to the target.

Holyoak and Thagard (1989) proposed a connectionist alternative in which a mapping is found by satisfying several soft constraints, namely structural consistency, semantic similarity and pragmatic importance. The theories differ in how they treat the role of goals and meaning, but agree on the importance of relational structure.

## Processing and development
Analogy requires retrieval of a suitable source, mapping, inference and often schema abstraction. Retrieval is strongly influenced by surface similarity, so people often fail to notice structurally appropriate sources spontaneously. In a classic study Gick and Holyoak (1980) found that participants given a related story solved a medical radiation problem far more often once they were prompted to use the story. Relational mapping places demands on [[working_memory]] and on prefrontal systems, and it develops through childhood.

## Relation to Active Inference
The connection to Active Inference is an analogy of its own kind, not a derivation. Analogical transfer can be read as reuse of a generative model: structure learned in one domain provides a prior over the structure of a new domain, so fewer observations are needed to infer its latent states. Work on structure learning and on hierarchical models, in which abstract higher levels are shared across contexts, supplies a candidate formal account (see [[category_learning]], [[hierarchical_processing]]). No standard Active Inference model of structure mapping has been established, and the claim remains a programme for research.

## Related concepts
- [[inductive_reasoning]] — generalisation, of which analogy is a case-based variant
- [[abductive_reasoning]] — generating explanatory hypotheses, often analogically
- [[category_learning]] — learning abstract structure from examples
- [[reasoning_problem_solving]] — analogy as a problem-solving strategy
- [[working_memory]] — capacity limits on relational mapping

## References
- Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy. Cognitive Science, 7, 155-170.
- Gick, M. L., & Holyoak, K. J. (1980). Analogical problem solving. Cognitive Psychology, 12, 306-355.
- Holyoak, K. J., & Thagard, P. (1989). Analogical mapping by constraint satisfaction. Cognitive Science, 13, 295-355.
