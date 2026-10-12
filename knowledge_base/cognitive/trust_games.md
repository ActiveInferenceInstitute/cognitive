---
title: Trust Games
description: Experimental economic games, especially the investment game of Berg, Dickhaut and McCabe, that measure willingness to trust and reciprocate, with their structure, findings and modelling.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - trust
  - game_theory
  - behavioural_economics
semantic_relations:
  - type: relates
    links:
      - "[[game_theory]]"
      - "[[trust_building]]"
      - "[[cooperation]]"
      - "[[social_preferences]]"
      - "[[theory_of_mind]]"
---

# Trust Games

## Overview
Trust games are laboratory tasks in which one player can make a costly move that benefits both players only if the other player reciprocates. They are used to measure trust, defined behaviourally as the willingness to make oneself vulnerable to another's choice, and trustworthiness, the tendency to honour that trust. The best known version is the investment game introduced by Berg, Dickhaut and McCabe (1995).

## The investment game
An investor receives an endowment and chooses how much of it to send to a trustee. The amount sent is multiplied, commonly by three, before the trustee receives it. The trustee then chooses how much to return. If both players care only about their own money, the trustee has no incentive to return anything, and the investor, anticipating this, should send nothing, so the standard prediction by backward induction is zero. In many experiments investors send a substantial share of the endowment and trustees return amounts that rise with what they received, interpreted as evidence of trust and positive reciprocity.

Variants include repeated games, in which reputation can develop, and games with a risk-only control in which the trustee is replaced by a random device, to separate trust from general risk attitude. Sent amounts are influenced by beliefs, social preferences, betrayal aversion and culture (see [[social_preferences]]). Neuroscience studies have used multi-round versions: King-Casas and colleagues (2005) reported brain activity in the caudate nucleus of the trustee that shifted over rounds from following to anticipating the partner's reciprocation. Intranasal oxytocin was reported to increase trusting transfers (Kosfeld et al., 2005), but later replications and analyses have produced mixed results, so the effect is contested.

## Relation to Active Inference
Trust can be modelled as a belief about the type or reliability of the partner, updated from observed returns. In Active Inference formulations, an agent holds a generative model that includes the partner's likely responses, and each round produces prediction errors that revise those beliefs and affect the policy chosen; repeated-game behaviour can then be fitted with preferences, precisions and belief parameters. This is a modelling approach used in computational psychiatry and social decision research, comparable in role to Bayesian and reinforcement learning models, and its distinctive advantage over them has not been settled.

## Related concepts
- [[game_theory]] — formal analysis that gives the zero-transfer prediction
- [[trust_building]] — how trust develops through repeated interaction
- [[trust_repair]] — recovery after a violation, often studied with these games
- [[cooperation]] — wider category of mutually beneficial behaviour
- [[oxytocin]] — hormone studied in relation to trusting behaviour
- [[theory_of_mind]] — modelling the partner's intentions

## References
- Berg, J., Dickhaut, J., & McCabe, K. (1995). Trust, reciprocity, and social history. Games and Economic Behavior, 10, 122-142.
- King-Casas, B., Tomlin, D., Anen, C., Camerer, C. F., Quartz, S. R., & Montague, P. R. (2005). Getting to know you: Reputation and trust in a two-person economic exchange. Science, 308, 78-83.
- Kosfeld, M., Heinrichs, M., Zak, P. J., Fischbacher, U., & Fehr, E. (2005). Oxytocin increases trust in humans. Nature, 435, 673-676.
