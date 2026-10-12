---
title: Nowak's Five Rules for the Evolution of Cooperation
description: Martin Nowak's classification of five mechanisms, kin selection, direct and indirect reciprocity, network reciprocity and group selection, with the benefit-to-cost conditions under which each favours cooperation.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - evolutionary_game_theory
  - cooperation
  - social_evolution
semantic_relations:
  - type: relates
    links:
      - "[[prisoners_dilemma]]"
      - "[[direct_reciprocity]]"
      - "[[indirect_reciprocity]]"
      - "[[evolutionary_game_theory]]"
      - "[[cooperation]]"
---

# Nowak's Five Rules for the Evolution of Cooperation

## Overview
Natural selection favours defection in a simple one-shot social dilemma, yet cooperation is widespread in biological and human societies. Nowak (2006) reviewed the theoretical literature and argued that mechanisms for the evolution of cooperation fall into five classes. For each he derived a condition, stated in terms of the benefit $b$ that a cooperator confers on a recipient and the cost $c$ it pays, under which cooperation is favoured over defection in a simple donation game. The paper does not claim that these are the only possible mechanisms.

## The five mechanisms
- **Kin selection.** Cooperation is directed at genetic relatives. With coefficient of relatedness $r$, the condition is $r>c/b$, which is Hamilton's rule.
- **Direct reciprocity.** The same pair of individuals meets repeatedly. With $w$ the probability of another round, cooperation can be favoured when $w>c/b$ (see [[direct_reciprocity]] and [[tit_for_tat]]).
- **Indirect reciprocity.** Cooperation depends on reputation, with helpful individuals more likely to be helped by third parties. With $q$ the probability of knowing another individual's reputation, the condition is $q>c/b$ ([[indirect_reciprocity]]).
- **Network reciprocity.** Individuals interact on a graph, so cooperators can form clusters. For a regular graph of degree $k$, the condition is $b/c>k$.
- **Group selection.** Competition between groups can favour cooperative groups. With groups of maximum size $n$ and $m$ groups, the condition is $b/c>1+n/m$.

All five conditions are of the form that the benefit-to-cost ratio must exceed a threshold set by the mechanism, and each reduces to the requirement that cooperators interact preferentially with other cooperators, or their descendants, more often than the population average. They derive from simplified models, such as large populations in the weak-selection limit, and the thresholds should be read with those assumptions in mind.

## Relation to Active Inference
The link is thematic rather than a derivation. The five rules are results of evolutionary dynamics on populations ([[replicator_dynamics]], [[population_games]]), not of inference by individual agents. Work on multi-agent Active Inference asks how agents that model one another's beliefs can sustain cooperation, and the reciprocity mechanisms are natural benchmarks for such models, but no standard mapping from Nowak's conditions to free energy minimisation exists.

## Related concepts
- [[prisoners_dilemma]] — the dilemma that these mechanisms resolve
- [[direct_reciprocity]] — repeated interaction between the same agents
- [[indirect_reciprocity]] — cooperation conditioned on reputation
- [[evolutionary_game_theory]] — the modelling framework
- [[cooperation]] — the cognitive and social phenomenon
- [[game_theory]] — the general theory of strategic interaction

## References
- Nowak, M. A. (2006). Five rules for the evolution of cooperation. Science, 314, 1560-1563.
- Nowak, M. A. (2006). Evolutionary Dynamics: Exploring the Equations of Life. Harvard University Press.
