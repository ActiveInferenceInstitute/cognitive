---
title: Direct Reciprocity
description: A mechanism for the evolution of cooperation in which repeated encounters between the same two individuals let returned help be rewarded, analysed through the repeated prisoner's dilemma.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - cooperation
  - evolutionary_game_theory
  - repeated_games
semantic_relations:
  - type: relates
    links:
      - "[[cooperation]]"
      - "[[indirect_reciprocity]]"
      - "[[prisoners_dilemma]]"
      - "[[tit_for_tat]]"
      - "[[evolutionary_game_theory]]"
---

# Direct Reciprocity

## Overview
Direct reciprocity is cooperation sustained by repeated interaction between the same pair of individuals: one helps now because the partner is likely to help back later. Trivers (1971) introduced it in biology as reciprocal altruism. In one-shot interactions with a prisoner's dilemma payoff structure, defection is the dominant choice; repetition changes the incentives because future returns depend on present conduct.

## The repeated prisoner's dilemma
In each round a player either cooperates, paying a cost $c$ to give the partner a benefit $b>c$, or defects. If the game continues to another round with probability $w$, the expected number of rounds is $1/(1-w)$. Always-defect is a strategy that cannot be invaded by a rare cooperator, but a reciprocating strategy can also be stable. For tit-for-tat, which cooperates first and then copies the partner's last move, cooperation is favoured when $w>c/b$ (Nowak, 2006). The general principle that long enough horizons support many outcomes is formalised by folk theorems for repeated games.

Axelrod and Hamilton (1981) and Axelrod's computer tournaments showed that simple, nice, retaliatory and forgiving strategies perform well against a diverse field. Later work found that tit-for-tat is vulnerable to errors, since a single mistaken defection can trigger a long retaliation cycle, and that strategies such as generous tit-for-tat or win-stay, lose-shift can do better in noisy settings.

## Conditions and limits
Direct reciprocity needs recognition of partners, memory of past moves and a reasonable chance of meeting again. It works best in small groups. With many players, tracking every pairwise history becomes costly, which motivates indirect reciprocity based on reputation.

## Relation to Active Inference
There is no derivation of reciprocity from the free energy principle. The link is an analogy and a modelling opportunity: an Active Inference agent playing a repeated game can maintain beliefs about a partner's policy and choose actions whose expected outcomes reflect that model, which is a form of the theory of mind required to track a partner's history. Such simulations exist as demonstrations, but they illustrate the framework rather than establish the evolutionary results above.

## Related concepts
- [[cooperation]] — the broader topic of which reciprocity is one mechanism
- [[indirect_reciprocity]] — cooperation sustained by reputation among third parties
- [[prisoners_dilemma]] — the game used to analyse reciprocal strategies
- [[tit_for_tat]] — the best-known reciprocating strategy
- [[game_theory]] — the mathematical framework for strategic interaction

## References
- Trivers, R. L. (1971). The evolution of reciprocal altruism. Quarterly Review of Biology, 46, 35-57.
- Axelrod, R., & Hamilton, W. D. (1981). The evolution of cooperation. Science, 211, 1390-1396.
- Nowak, M. A. (2006). Five rules for the evolution of cooperation. Science, 314, 1560-1563.
