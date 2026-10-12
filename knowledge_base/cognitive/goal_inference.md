---
title: Goal Inference
description: Inferring the goals or intentions behind another agent's observed actions, its developmental origins, the inverse planning formulation, and its close fit with Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - social_cognition
  - intention
  - inverse_planning
semantic_relations:
  - type: relates
    links:
      - "[[social_cognition]]"
      - "[[theory_of_mind]]"
      - "[[action_recognition]]"
      - "[[mirror_system]]"
      - "[[joint_action]]"
---

# Goal Inference

## Overview
Goal inference is the process of working out what an agent is trying to achieve from observation of what it does. Humans readily describe movements in terms of purposes: a hand moving towards a cup is seen as reaching for it, even if the cup is never touched. The capacity underlies imitation, cooperation, teaching and language learning, and it is considered a basic component of [[social_cognition]] and, in more elaborate forms, of [[theory_of_mind]].

## Development
Infants appear to interpret actions as goal-directed from early in the first year. Gergely and Csibra (2003) proposed that infants apply a teleological stance, according to which an action is judged rational if it is an efficient means of achieving a goal in the given situation. In their studies infants looked longer at actions that were inefficient given the constraints than at efficient ones. Because the proposal has been debated, particularly on the roles of experience and of minimal cues such as self-propelled motion, the strength of the claims for early rationality varies across studies.

## Computational formulation
Baker, Saxe and Tenenbaum (2009) framed action understanding as inverse planning. An observer assumes that the agent chooses actions that approximately maximise expected utility given its beliefs and desires, and uses Bayes' rule to infer the goals that best explain the observed trajectory:

$$P(g \mid a_{1:t}, s_{1:t}) \propto P(a_{1:t} \mid g, s_{1:t})\, P(g)$$

Their model predicted human judgements of the goals of an agent moving in simple environments. It builds on the idea that the observer holds a model of the agent as a planner.

## Neural accounts
Mirror-system accounts propose that observed actions are mapped onto the observer's own motor repertoire. Kilner, Friston and Frith (2007) recast this as predictive coding: the goal at the top of a hierarchy generates predictions of the kinematics of the observed action, and the best-fitting goal is the one that minimises prediction error (see [[mirror_system]]).

## Relation to Active Inference
The connection is strong and direct in formulation. Goal inference is Bayesian inference about another agent's generative model and preferences, treating the other as an Active Inference agent whose policies minimise expected free energy. Applications include modelling joint action and communication (see [[joint_action]]), and the models have been tested chiefly in simulation.

## Related concepts
- [[theory_of_mind]] — attributing mental states to others
- [[action_recognition]] — identifying observed actions
- [[mirror_system]] — circuits for observed and executed actions
- [[joint_action]] — coordination that depends on inferring partners' goals
- [[social_cognition]] — the wider domain

## References
- Gergely, G., & Csibra, G. (2003). Teleological reasoning in infancy: The naive theory of rational action. Trends in Cognitive Sciences, 7, 287-292.
- Kilner, J. M., Friston, K. J., & Frith, C. D. (2007). Predictive coding: An account of the mirror neuron system. Cognitive Processing, 8, 159-166.
- Baker, C. L., Saxe, R., & Tenenbaum, J. B. (2009). Action understanding as inverse planning. Cognition, 113, 329-349.
