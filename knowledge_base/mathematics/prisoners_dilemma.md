---
title: Prisoner's Dilemma
description: The two-player game in which individually rational defection yields a worse outcome than mutual cooperation, its payoff conditions, repeated versions, and its role in the study of cooperation.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - game_theory
  - cooperation
  - social_dilemma
semantic_relations:
  - type: relates
    links:
      - "[[game_theory]]"
      - "[[tit_for_tat]]"
      - "[[cooperation]]"
      - "[[direct_reciprocity]]"
      - "[[nowak_five_rules]]"
wikidata: Q41693
---

# Prisoner's Dilemma

## Overview
The prisoner's dilemma is a two-player game in which each player chooses to cooperate (C) or defect (D). Defection is the better reply to either choice of the opponent, yet mutual cooperation gives both players more than mutual defection. The game was formulated at RAND in 1950 by Merrill Flood and Melvin Dresher, and the prison story that gives it its name is attributed to Albert Tucker. It serves as the standard abstract model of a social dilemma.

## Payoff structure
Let $R$ be the reward for mutual cooperation, $P$ the punishment for mutual defection, $T$ the temptation to defect against a cooperator, and $S$ the sucker's payoff. The game is a prisoner's dilemma when
$$T>R>P>S.$$
Then D strictly dominates C, so (D, D) is the unique Nash equilibrium, although both players prefer (C, C). For repeated play, the condition $2R>T+S$ is usually added, so that alternating between exploiting and being exploited does not beat steady mutual cooperation. A common numerical instance is $(T,R,P,S)=(5,3,1,0)$. In the donation-game form, a cooperator pays cost $c$ to give benefit $b>c$ to the partner, so $T=b$, $R=b-c$, $P=0$ and $S=-c$.

## Repeated play
- **Finite, known horizon.** Backward induction implies defection in every round.
- **Indefinite horizon.** If a further round occurs with probability $w$, many strategies, including conditional cooperation, form equilibria for sufficiently large $w$ (the folk theorem for repeated games).
- **Tournaments.** Axelrod's computer tournaments found that the simple reciprocating strategy [[tit_for_tat]] performed well, which stimulated extensive work on how cooperation can arise among self-interested agents (Axelrod & Hamilton, 1981).
- **Spatial and population versions.** Placing players on a graph or in a population changes the outcome ([[nowak_five_rules]], [[population_games]]).

## Relation to Active Inference
The prisoner's dilemma is a test case for models of social cognition rather than a part of Active Inference theory. In multi-agent settings, agents that each maintain a generative model of the other's policy can be simulated in repeated games, and whether cooperation emerges depends on the prior preferences and beliefs assumed. Such studies are exploratory, and no general result links free energy minimisation to cooperative outcomes in the dilemma. The related empirical literature on [[cooperation]] in humans and animals uses the game extensively.

## Related concepts
- [[game_theory]] — the framework in which the game is analysed
- [[tit_for_tat]] — the best-known strategy for the repeated game
- [[direct_reciprocity]] — cooperation sustained by repeated encounters
- [[nowak_five_rules]] — mechanisms that overcome the dilemma
- [[evolutionary_game_theory]] — evolutionary analysis of the game
- [[cooperation]] — empirical and cognitive aspects

## References
- Axelrod, R., & Hamilton, W. D. (1981). The evolution of cooperation. Science, 211, 1390-1396.
- Axelrod, R. (1984). The Evolution of Cooperation. Basic Books.
- Nowak, M. A. (2006). Evolutionary Dynamics: Exploring the Equations of Life. Harvard University Press.
