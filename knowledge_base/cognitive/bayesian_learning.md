---
title: Bayesian Learning
description: Learning treated as Bayesian updating of beliefs about model parameters from data, with conjugate worked examples and its Dirichlet-count form in Active Inference.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - learning
  - bayesian
  - parameters
semantic_relations:
  - type: relates
    links:
      - "[[bayesian_inference]]"
      - "[[bayesian_brain]]"
      - "[[active_inference]]"
---

# Bayesian Learning

## Overview
Bayesian learning treats the unknown parameters of a model as random variables and updates their distribution as data arrive. It is the same calculus as [[bayesian_inference]] about hidden states, applied to quantities that change slowly or not at all. Prior knowledge enters as a prior over parameters; evidence enters through the likelihood; the posterior after one batch serves as the prior for the next.

## Formalism
For parameters $\theta$ and data $D$, $p(\theta \mid D) \propto p(D \mid \theta)\,p(\theta)$. Sequential updating gives the same posterior as processing all data at once, provided observations are conditionally independent given $\theta$.

Conjugate priors give closed-form updates. For a Bernoulli outcome with a Beta$(\alpha,\beta)$ prior, observing $s$ successes in $n$ trials yields a Beta$(\alpha+s,\ \beta+n-s)$ posterior. Starting from Beta$(1,1)$ and observing 7 successes in 10 trials gives Beta$(8,4)$, with posterior mean $8/12 \approx 0.667$. For categorical outcomes the analogue is the Dirichlet distribution, whose parameters act as pseudo-counts: each observation increments the count of the observed category. The posterior narrows as counts grow, so the effective learning rate falls with experience, an important difference from fixed-step gradient rules.

Model selection extends the idea one level up: the marginal likelihood $p(D \mid m)$ of each candidate model $m$ weighs fit against complexity (see [[model_selection]]).

## Role in Active Inference
In discrete-state Active Inference, the likelihood mapping $A$, transition mappings $B$ and initial-state prior $D$ are given Dirichlet priors with concentration parameters $a$, $b$, $d$. After each time step, learning increments these counts in proportion to the posterior beliefs about states and the observed outcomes, for example $a \leftarrow a + \eta\, o \otimes s$ (Da Costa et al., 2020). Learning is therefore a slower stage of the same free-energy minimisation that updates state beliefs, and it supplies a quantity (expected information gain about parameters) that can motivate exploratory policies (see [[active_learning]], [[epistemic_value]]). This is a derivation within the discrete formulation; whether biological synapses implement counting of this form is a separate and open empirical question (see [[biological_plausibility]]).

## Related concepts
- [[bayesian_brain]] — the hypothesis that neural systems approximate Bayesian inference
- [[belief_updating]] — updating of state beliefs on the fast timescale
- [[belief_initialization]] — choice of priors over states and parameters
- [[structure_learning]] — learning model structure rather than parameters
- [[prior_beliefs]] — the role of priors in constraining learning

## References
- Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2020). Active inference on discrete state-spaces: a synthesis. Journal of Mathematical Psychology, 99, 102447.
