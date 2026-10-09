---
title: Knowledge Representation
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - artificial_intelligence
  - representation
  - knowledge
semantic_relations:
  - type: relates
    links:
      - "[[generative_model]]"
      - "[[knowledge_graphs]]"
      - "[[semantic_memory]]"
      - "[[active_inference]]"
---

# Knowledge Representation

## Overview
Knowledge representation (KR) is the study of how information about the world can be encoded so that a system can use it to reason and act. In artificial intelligence, KR and reasoning is a subfield concerned with formalisms that are expressive enough to describe a domain and tractable enough to compute with. In cognitive science the same question is asked of minds: in what format do people store concepts and beliefs?

## Mechanism
Davis, Shrobe, and Szolovits described a representation as a surrogate for the world, a set of ontological commitments, a fragmentary theory of intelligent reasoning, a medium for efficient computation, and a medium of human expression. Main families include:

- **Logic-based**: propositional and first-order logic, and description logics underlying OWL ontologies, which support sound inference but can be brittle with uncertainty.
- **Structured symbolic**: semantic networks (Quillian), frames (Minsky), scripts, and production rules.
- **Graph-based**: knowledge graphs of entities and typed relations ([[knowledge_graphs]]), and ontologies ([[cognitive_ontology]]).
- **Probabilistic**: Bayesian networks and Markov random fields (Pearl), which represent uncertain dependencies.
- **Distributed**: vector embeddings and neural network weights, where knowledge is implicit in learned parameters and not directly inspectable.

A central tradeoff, formalised by Levesque and Brachman, is expressiveness versus tractability: richer languages make reasoning harder, sometimes undecidable. The symbolic and distributed traditions differ in transparency, handling of uncertainty, and ability to learn from data, and hybrid neurosymbolic systems try to combine them. In human memory, theories of [[semantic_memory]] and conceptual structure address related questions ([[knowledge_organization]]).

## Role in Active Inference
In Active Inference, the knowledge of an agent is its generative model: a probabilistic specification of how hidden states generate observations (likelihood), how states evolve (transitions), what outcomes are preferred, and prior beliefs about states. This is a probabilistic, factorised representation of the Bayesian-network type, and it is learned by updating the model's parameters ([[generative_model]], [[belief_updating]]). In this repository's implementations the model is a set of matrices (see [[matrix_specifications]]). The choice of state-space structure and its abstraction is therefore a KR decision ([[state_space_abstraction]]). Active Inference does not itself prescribe how the structure is found, and richly structured, symbolic, or relational knowledge is harder to express in a discrete matrix model than in the formalisms above.

## Related concepts
- [[generative_model]] — the agent's representation in Active Inference
- [[knowledge_graphs]] — graph-structured knowledge
- [[semantic_memory]] — human conceptual knowledge
- [[knowledge_organization]] — organisation of stored knowledge
- [[cognitive_ontology]] — an ontology of cognitive concepts
- [[state_space_abstraction]] — choosing the model's representational structure

## References
- Davis, R., Shrobe, H., & Szolovits, P. (1993). What is a knowledge representation? *AI Magazine*.
- Levesque, H. J., & Brachman, R. J. (1987). Expressiveness and tractability in knowledge representation and reasoning. *Computational Intelligence*.
- Brachman, R. J., & Levesque, H. J. (2004). *Knowledge Representation and Reasoning*. Morgan Kaufmann.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann.
