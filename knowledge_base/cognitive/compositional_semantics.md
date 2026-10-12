---
title: Compositional Semantics
description: The principle that the meaning of a complex expression is determined by the meanings of its parts and how they combine, formal approaches, challenges, and links to hierarchical generative models.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - semantics
  - language
  - compositionality
semantic_relations:
  - type: relates
    links:
      - "[[language_comprehension]]"
      - "[[semantic_memory]]"
      - "[[syntactic_parsing]]"
      - "[[thematic_roles]]"
      - "[[pragmatics]]"
---

# Compositional Semantics

## Overview
Compositional semantics studies how the meaning of a phrase or sentence is built from the meanings of its words and the syntactic structure that combines them. It rests on the principle of compositionality, usually credited to Frege, that the meaning of a complex expression is a function of the meanings of its parts and the way they are put together. The principle is often invoked to explain productivity, the ability to understand an unbounded number of new sentences, and systematicity, the fact that understanding "John loves Mary" goes with understanding "Mary loves John".

## Formal approaches
- **Truth-conditional semantics.** Montague (1973) treated natural language as interpreted in a model-theoretic way, assigning each syntactic rule a corresponding semantic operation. Heim and Kratzer (1998) present a standard textbook version for generative grammar, in which meanings are typed functions and composition proceeds mainly by function application.
- **Examples.** The meaning of a verb phrase can be a function from individuals to truth values, which takes a subject to yield a sentence meaning. Quantifiers such as "every" are functions over sets or properties (see [[predicate_logic]]).
- **Distributional approaches.** Words can be represented by vectors that capture patterns of use, and composition by operations on vectors or by neural networks (see [[distributed_representation]]). Whether such systems achieve human-like systematicity is debated (Fodor and Pylyshyn, 1988).

## Challenges
Idioms such as "kick the bucket" have meanings that are not predictable from the parts. Context affects the interpretation of many expressions, for example "ready" or "fast", and the division of labour between literal meaning and speaker meaning is a matter of debate (see [[pragmatics]]). Coercion, metaphor and polysemy also require flexible composition. Psycholinguistic work shows that people integrate word meaning with context incrementally and sometimes build interpretations that are good enough but not fully compositional (see [[sentence_processing]]).

## Relation to Active Inference
The link is weak and analogical. Comprehension has been described as hierarchical prediction, in which the listener's expectations about upcoming words and meaning are tested against input, and neural responses such as the N400 have been interpreted in terms of prediction (Kuperberg and Jaeger, 2016; see [[predictive_processing]]). Active Inference models of language use hierarchical generative models in which higher levels carry slowly changing context and lower levels carry words or sounds, which parallels the idea of composition. No account has derived compositional semantic principles from the free energy principle.

## Related concepts
- [[syntactic_parsing]] — building the structure on which composition operates
- [[thematic_roles]] — roles linking predicates to arguments
- [[semantic_networks]] — representations of relations among concepts
- [[noun_phrases]] — phrase type often analysed in composition
- [[language_comprehension]] — the processing of meaning in context

## References
- Montague, R. (1973). The proper treatment of quantification in ordinary English. In J. Hintikka, J. Moravcsik, & P. Suppes (Eds.), Approaches to natural language (pp. 221-242). Reidel.
- Heim, I., & Kratzer, A. (1998). Semantics in generative grammar. Blackwell.
- Fodor, J. A., & Pylyshyn, Z. W. (1988). Connectionism and cognitive architecture: A critical analysis. Cognition, 28, 3-71.
- Kuperberg, G. R., & Jaeger, T. F. (2016). What do we mean by prediction in language comprehension? Language, Cognition and Neuroscience, 31, 32-59.
