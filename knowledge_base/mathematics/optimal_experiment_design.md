---
title: Optimal Experiment Design
description: "Choosing experiments to maximise information about parameters, covering alphabetic optimality criteria, Bayesian expected information gain, and the link to epistemic value in Active Inference."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - statistics
  - information_theory
  - exploration
semantic_relations:
  - type: relates
    links:
      - "[[information_gain]]"
      - "[[uncertainty_resolution]]"
      - "[[epistemic_value]]"
      - "[[fisher_information]]"
---

# Optimal Experiment Design

## Overview
Optimal experiment design chooses controllable settings $d$ of an experiment (sample locations, stimuli, queries) so that the data $y$ are as informative as possible about unknown parameters $\theta$. Classical design works with the Fisher information matrix; Bayesian design works with expected reduction in posterior uncertainty.

## Criteria
**Alphabetic optimality (classical).** For a linear model with information matrix $M(d)$, D-optimality maximises $\det M$ (smallest confidence ellipsoid volume), A-optimality minimises $\operatorname{tr}M^{-1}$ (average variance), and E-optimality maximises $\lambda_{\min}(M)$ (worst-case direction). These depend on the unknown $\theta$ when the model is nonlinear, which motivates local or Bayesian versions.

**Bayesian expected information gain.** Lindley (1956) proposed scoring a design by the expected KL divergence from prior to posterior:
$$\mathrm{EIG}(d)=\mathbb E_{p(y\mid d)}\big[D_{KL}\big(p(\theta\mid y,d)\,\|\,p(\theta)\big)\big]=I(\theta;y\mid d),$$
which is the mutual information between parameters and data. Choosing $d^*=\arg\max_d \mathrm{EIG}(d)$ is Bayes-optimal under the log-score utility. Sequential design repeats this after each observation (adaptive or active learning).

**Computation.** EIG is a nested expectation and generally needs approximation (nested Monte Carlo, variational bounds on mutual information; see [[mutual_information_estimators]]). In the discrete case it equals $H[Q(o)]-\mathbb E_Q H[P(o\mid s)]$.

## Role in Active Inference
The connection is close and mathematical. The epistemic term of [[expected_free_energy]] is the expected information gain about hidden states (or about parameters), the same mutual information as Lindley's criterion. With the repo convention $G=\text{risk}+\text{ambiguity}$, one can rewrite $G=-\mathbb E_Q[\ln P(o\mid C)]-I(s;o)$, so minimising $G$ trades preference satisfaction against information gain. I confirmed this identity numerically on a random discrete model: both sides gave 1.28995. Active Inference agents are therefore sequential Bayesian experimental designers whose utility also includes preferences. They differ from textbook design in using approximate posteriors and a finite policy set rather than an optimisation over a continuous design space.

## Related concepts
- [[information_gain]] — the quantity maximised by Bayesian design
- [[epistemic_value]] — its name within Active Inference
- [[exploration_exploitation]] — the preference-versus-information trade-off
- [[active_learning]] — query selection in machine learning
- [[uncertainty_resolution]] — reduction of entropy as the design goal

## References
- Lindley, D. V. (1956). On a measure of the information provided by an experiment. Annals of Mathematical Statistics.
- Chaloner, K. and Verdinelli, I. (1995). Bayesian experimental design: a review. Statistical Science.
