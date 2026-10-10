---
title: Causal Inference
description: Methods for estimating the effects of interventions from data using causal graphs and potential outcomes, including the do-operator, back-door adjustment, and the relation of causal reasoning to Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - causality
  - graphical_models
  - statistics
semantic_relations:
  - type: relates
    links:
      - "[[bayesian_networks]]"
      - "[[d_separation]]"
      - "[[conditional_independence]]"
      - "[[graphical_models]]"
---

# Causal Inference

## Overview
Causal inference asks what would happen under an intervention, which is a different question from what is associated with what in observed data. Two frameworks dominate: structural causal models with graphs (Pearl) and potential outcomes (Rubin). They are formally compatible and differ mainly in notation and in how assumptions are stated.

## Formalism
A structural causal model assigns each variable a function of its parents and independent noise, $X_i=f_i(\mathrm{pa}_i,U_i)$, with the parent structure drawn as a directed acyclic graph (see [[causal_structure]]). Pearl describes three rungs of reasoning: association, $P(y\mid x)$; intervention, $P(y\mid \mathrm{do}(x))$; and counterfactuals, which ask what would have happened for this particular case.

The do-operator replaces the equation for $X$ by a constant, deleting the arrows into $X$. If a set $Z$ blocks every back-door path from $X$ to $Y$ ([[d_separation]]) and contains no descendants of $X$, then
$$P(y\mid \mathrm{do}(x))=\sum_z P(y\mid x,z)\,P(z).$$
Randomised experiments achieve the same by design. Other routes are front-door adjustment, instrumental variables and do-calculus, the latter complete for identification in graphs (Shpitser and Pearl, 2006). Structure can also be learned from conditional independence tests, up to a Markov equivalence class (Spirtes, Glymour and Scheines).

Correlation without causation is the standard failure: a common cause $Z$ makes $X$ and $Y$ dependent although neither affects the other, and conditioning on a collider creates dependence between its causes.

## Role in Active Inference
Two links are well grounded. First, an agent's actions act as interventions on its environment: choosing a policy sets a variable by fiat, so the expected-free-energy comparison of policies is a comparison of predicted outcomes under intervention, which requires the model to have a causal reading. Second, the generative models used in Active Inference are directed graphical models, and dynamic causal modelling uses experimental inputs as interventions to infer effective connectivity. What does not follow is that any fitted generative model is causally correct: Bayesian inference on a graph recovers conditional dependence, and the causal interpretation requires structural assumptions or interventional data. Active exploration, with epistemic value driving the agent to try actions that disambiguate hypotheses, is one way the framework can learn causal structure.

## Related concepts
- [[bayesian_networks]] — directed graphs on which causal models are built
- [[d_separation]] — graphical criterion for back-door sets and independence
- [[conditional_independence]] — the statistical signature used for structure learning
- [[epistemic_value]] — drive to take informative actions, close to experimental design
- [[structure_learning]] — inferring graph structure from data
- [[probabilistic_graphical_models]] — wider family of graph-based models

## References
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge University Press.
- Shpitser, I. and Pearl, J. (2006). Identification of joint interventional distributions in recursive semi-Markovian causal models. *Proceedings of the 21st National Conference on Artificial Intelligence (AAAI)*.
- Spirtes, P., Glymour, C. and Scheines, R. (2000). *Causation, Prediction, and Search*, 2nd ed. MIT Press.
