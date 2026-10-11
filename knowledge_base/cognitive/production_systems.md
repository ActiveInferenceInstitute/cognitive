---
title: Production Systems
description: Rule-based models of cognition built from condition-action pairs acting on a working memory, covering the recognise-act cycle, architectures such as ACT-R and Soar, and contrast with probabilistic accounts.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - cognitive_architecture
  - rules
  - symbolic
semantic_relations:
  - type: relates
    links:
      - "[[cognitive_architecture]]"
      - "[[reasoning_problem_solving]]"
      - "[[working_memory]]"
      - "[[procedural_memory]]"
      - "[[hierarchical_processing]]"
---

# Production Systems

## Overview
A production system models cognition as a set of if-then rules, called productions, that operate on a store of current information. Each production has a condition that tests the contents of working memory and an action that modifies it or triggers behaviour. Newell and Simon (1972) used production systems to model human problem solving, and the formalism became a foundation of symbolic cognitive science.

## The recognise-act cycle
A production system has three parts: a working memory holding the current facts and goals, a long-term store of productions, and an interpreter. The interpreter repeats a cycle:
1. **Match**: find all productions whose conditions are satisfied by working memory.
2. **Conflict resolution**: choose one of them, using criteria such as recency, specificity or learned utility.
3. **Act**: execute the chosen action, changing working memory or the environment.

Because only one or a few productions fire at a time, behaviour unfolds as a sequence of small steps, which gives a natural account of serial processing and allows predictions of reaction times from the number of cycles. Complex skills arise from the interaction of many simple rules.

## Architectures
Several cognitive architectures extend the basic scheme (see [[cognitive_architecture]]). ACT (in its ACT* form, described by Anderson, 1983) distinguishes declarative memory, holding facts, from procedural memory, holding productions, and models skill learning as the compilation of declarative knowledge into rules (see [[procedural_memory]]). Soar (Laird, Newell, & Rosenbloom, 1987) casts all activity as problem-space search and learns new productions by chunking the results of subgoals. Both architectures have been used to fit behavioural data across many tasks, and later versions of ACT-R also map components to brain regions.

## Relation to Active Inference
Production systems and Active Inference sit at different levels of description. Productions are discrete, symbolic and deterministic or utility-weighted, whereas Active Inference uses probabilistic generative models and selects policies by expected free energy. A loose correspondence exists in that habitual policies in a discrete model act like condition-action rules, and a learned prior over policies selects actions given the current state (see [[policy_selection]]). Hierarchical models also yield rule-like behaviour at higher levels (see [[hierarchical_processing]]). This resemblance is an analogy, and no standard derivation of production-system architectures from the free energy principle exists.

## Related concepts
- [[cognitive_architecture]] — the broader class of unified models
- [[reasoning_problem_solving]] — the domain modelled by early production systems
- [[procedural_memory]] — memory for skills and rules
- [[working_memory]] — the store on which productions operate
- [[hierarchical_processing]] — layered organisation of rules and goals

## References
- Newell, A., & Simon, H. A. (1972). Human problem solving. Prentice-Hall.
- Anderson, J. R. (1983). The architecture of cognition. Harvard University Press.
- Laird, J. E., Newell, A., & Rosenbloom, P. S. (1987). SOAR: An architecture for general intelligence. Artificial Intelligence, 33, 1-64.
