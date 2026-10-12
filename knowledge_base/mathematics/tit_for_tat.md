---
title: Tit for Tat
description: The reciprocating strategy for the iterated prisoner's dilemma that cooperates first and then copies the partner's previous move, with its success in tournaments, its limits under noise, and its successors.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - game_theory
  - reciprocity
  - repeated_games
semantic_relations:
  - type: relates
    links:
      - "[[prisoners_dilemma]]"
      - "[[direct_reciprocity]]"
      - "[[nowak_five_rules]]"
      - "[[cooperation]]"
      - "[[evolutionary_game_theory]]"
wikidata: Q1367487
---

# Tit for Tat

## Overview
Tit for tat (TFT) is a strategy for the iterated [[prisoners_dilemma]]. It has two rules: cooperate in the first round, and thereafter do whatever the partner did in the previous round. It therefore rewards cooperation, punishes defection immediately and once only, and returns to cooperation as soon as the partner does. The strategy was submitted by Anatol Rapoport to the computer tournaments organised by Robert Axelrod around 1980, where it obtained the highest average score in both rounds, despite being the simplest entry in the first.

## Properties
Axelrod (1984) summarised the success of TFT through four properties shared by the high-scoring strategies.
- **Nice.** It never defects first.
- **Provocable.** It responds to defection at once, which discourages exploitation.
- **Forgiving.** It does not hold a grudge beyond the last move.
- **Clear.** Its behaviour is easy for a partner to recognise and predict.

In a game with an indefinite horizon, with continuation probability $w$, a pair of TFT players is a Nash equilibrium when $w$ is large enough relative to the payoffs, and the threshold depends on the temptation, reward and punishment values. Mutual TFT then yields the cooperative payoff $R$ each round.

## Limits and successors
- **Noise.** If moves are occasionally misperceived or misexecuted, a single error sets off an alternating sequence of retaliation between two TFT players, so average payoffs fall. Generous tit for tat cooperates after a defection with some probability, which breaks such cycles.
- **Win-stay, lose-shift.** Nowak and Sigmund (1993) showed in simulations that a strategy that repeats its move after a good payoff and switches after a bad one can outperform TFT in noisy evolving populations.
- **Evolutionary stability.** No pure strategy, including TFT, is evolutionarily stable in the repeated prisoner's dilemma (Boyd & Lorberbaum, 1987), and success depends on the mix of competing strategies.

## Relation to Active Inference
TFT is a policy with a one-step memory, and models of agents in repeated games can reproduce reciprocal behaviour when an agent infers its partner's policy and holds a preference for mutual cooperation. The relation is an illustrative modelling exercise, not a theoretical consequence of free energy minimisation. TFT is also a standard reference point for empirical work on reciprocity ([[cooperation]], [[direct_reciprocity]]).

## Related concepts
- [[prisoners_dilemma]] — the game in which the strategy is defined
- [[direct_reciprocity]] — the mechanism TFT implements
- [[nowak_five_rules]] — conditions under which reciprocity is favoured
- [[cooperation]] — empirical aspects of reciprocal behaviour
- [[evolutionary_game_theory]] — the evolutionary analysis of strategies
- [[game_theory]] — the surrounding theory

## References
- Axelrod, R., & Hamilton, W. D. (1981). The evolution of cooperation. Science, 211, 1390-1396.
- Axelrod, R. (1984). The Evolution of Cooperation. Basic Books.
- Boyd, R., & Lorberbaum, J. P. (1987). No pure strategy is evolutionarily stable in the repeated Prisoner's Dilemma game. Nature, 327, 58-59.
- Nowak, M., & Sigmund, K. (1993). A strategy of win-stay, lose-shift that outperforms tit-for-tat in the Prisoner's Dilemma game. Nature, 364, 56-58.
