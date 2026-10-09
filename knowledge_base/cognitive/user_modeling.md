---
title: User Modeling
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - cognitive
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/systems/human_computer_interaction]]"
      - "[[knowledge_base/cognitive/preference_learning]]"
      - "[[knowledge_base/systems/technology_ethics]]"
---

# User Modeling

## Overview

User modeling is the practice of building and maintaining a model of a person's goals, knowledge, preferences and state so a system can adapt to them. Classical approaches include rule-based profiles, collaborative filtering over many users, and Bayesian networks over a user's task state. Applying Active Inference means treating the user as an agent with a generative model and treating the system as another agent that infers the user's hidden state and chooses what to show. This is a research proposal: the framework gives a coherent formulation, but there is little deployed evidence that it outperforms standard Bayesian or learned user models.

## Main Ideas

**Inferring hidden user state.** What the system sees is clicks, timing, corrections and, in some settings, gaze or physiological signals. What it needs is a hidden goal, preference or level of expertise. Posing this as inference over a latent variable with an explicit observation model gives calibrated uncertainty, which matters because interface decisions are made under it.

**Preference learning.** Preferences correspond to the $C$ vector of log preferences over outcomes. Inferring a user's $C$ from their choices is an inverse problem akin to inverse reinforcement learning; observed choices are evidence for the user's priors ([[knowledge_base/cognitive/preference_learning]]). The inverse problem is under-determined, since different preferences and beliefs can produce the same choices, so informative priors or deliberate probing are needed.

**Cognitive load estimation.** Load can be inferred from performance (error rate, response time) or from physiological measures such as pupil diameter and heart-rate variability. These signals are noisy and respond to many causes besides mental effort, so they function as weak observations in a model and not as direct readings.

**Adaptive interfaces as epistemic action.** A system choosing between acting on its current best guess and asking a clarifying question faces the exploration-exploitation trade-off. Expected free energy scores both: the question has information value, the action has pragmatic value, and the right choice depends on how uncertain the system is and what a wrong guess costs the user ([[knowledge_base/cognitive/epistemic_foraging]]).

## Risks

Adaptive systems can create feedback loops, in which a model of the user shapes what the user sees and so narrows the data it later learns from. A new user gives little data (the cold-start problem). Behavioural and physiological profiles are sensitive personal data, which raises consent and retention questions treated in [[knowledge_base/systems/technology_ethics]]. Users also need to be able to inspect and correct the model.

## In This Repository

See [[knowledge_base/systems/human_computer_interaction]] for interface design, [[knowledge_base/cognitive/preference_learning]] for the $C$ matrix, and [[knowledge_base/cognitive/social_cognition]] for modelling other agents. The code tree has no user-modeling implementation. The pieces it does provide, a validated discrete generative model and a variational belief update, could serve as the inference core for a small user model, but no inverse-preference estimation exists.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/systems/human_computer_interaction]]
