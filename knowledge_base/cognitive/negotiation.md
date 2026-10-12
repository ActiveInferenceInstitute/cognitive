---
title: Negotiation
description: Negotiation is the process by which parties with partly conflicting interests reach a joint agreement, covering bargaining theory, behavioural findings, and multi-agent Active Inference modelling.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - social-cognition
  - bargaining
  - game-theory
semantic_relations:
  - type: relates
    links:
      - "[[cooperation]]"
      - "[[game_theory]]"
      - "[[theory_of_mind]]"
wikidata: Q202875
---

# Negotiation

## Overview
Negotiation is communication and exchange between two or more parties who share some interests and conflict on others, aimed at a joint decision that each prefers to no agreement. It is studied in game theory as bargaining, in psychology as social decision making and conflict resolution, and in artificial intelligence as multi-agent protocol design. The cited pages use it as a form of social problem solving, so this page emphasizes cognitive processes together with the formal theory.

## Formalism
**Nash bargaining.** For two parties with utilities $u_1, u_2$ and disagreement payoffs $d_1, d_2$, the Nash solution (Nash, 1950) selects the feasible agreement maximizing the product $(u_1 - d_1)(u_2 - d_2)$. Splitting one unit of a divisible good with $d_1 = 0$ and $d_2 = 0.2$ gives party 1 a share of $(1 + d_1 - d_2)/2 = 0.4$; improving one's outside option raises one's share.

**Alternating offers.** In Rubinstein's model (1982), players take turns proposing splits and discount delay; the unique subgame-perfect equilibrium gives an advantage to the first mover that shrinks as discounting vanishes.

**Behavioural findings.** People depart from these equilibria: in the ultimatum game (Güth, Schmittberger, and Schwarze, 1982) responders often reject low but positive offers, and proposers offer substantial shares, reflecting fairness norms and anticipation of rejection ([[social_norms]]). Practitioner frameworks distinguish positions from underlying interests and advocate integrative solutions that create value (Fisher and Ury, 1981).

**Cognitive demands.** Effective negotiators infer the other side's priorities and reservation values from their offers, adopt their perspective, and regulate emotion ([[perspective_taking]], [[emotion_regulation]]). Anchoring on first offers and framing effects influence outcomes.

## Role in Active Inference
In multi-agent Active Inference, each agent holds a generative model that includes beliefs about the other agent's preferences and policies, so an offer is chosen to minimize expected free energy given the predicted response, with reciprocal modelling producing nested inference ([[multi_agent_active_inference]], [[theory_of_mind]]). Shared generative models, or alignment of beliefs via communication, are proposed as the basis of agreement. Simulations exist for coordination and communication games, but established quantitative accounts of human negotiation behaviour in this framework are lacking; it is a modelling direction, not a validated theory.

## Related concepts
- [[cooperation]] — cooperative behaviour that negotiation can establish
- [[game_theory]] — formal theory of strategic interaction
- [[theory_of_mind]] — inference about others' beliefs and goals
- [[multi_agent_active_inference]] — multi-agent modelling in the framework
- [[social_norms]] — fairness norms shaping acceptable offers
- [[decision_making]] — individual choice component of negotiation

## References
- Nash, J. F. (1950). The bargaining problem. *Econometrica*, 18, 155-162.
- Rubinstein, A. (1982). Perfect equilibrium in a bargaining model. *Econometrica*, 50, 97-109.
- Güth, W., Schmittberger, R., & Schwarze, B. (1982). An experimental analysis of ultimatum bargaining. *Journal of Economic Behavior and Organization*, 3, 367-388.
- Fisher, R., & Ury, W. (1981). *Getting to Yes: Negotiating Agreement Without Giving In*. Houghton Mifflin.
