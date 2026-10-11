---
title: Statistical Modeling
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - statistics
  - modeling
  - overview
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/statistical_foundations]]"
      - "[[knowledge_base/mathematics/probabilistic_models]]"
      - "[[knowledge_base/mathematics/generative_models]]"
      - "[[knowledge_base/research/concepts/model_comparison]]"
aliases:
  - statistical_models
---

# Statistical Modeling

## Overview
Statistical modeling is the construction of a probabilistic description of data, fitting its free quantities, and checking whether it is adequate. It is the layer between raw measurements and conclusions, and in this repository it covers both the agent's internal generative model and the models used to analyse simulation outputs.

## Definition / Formalism
A statistical model is a family of distributions $\{p(y\mid\theta):\theta\in\Theta\}$, optionally with a prior $p(\theta)$. The usual cycle is:

1. Specify: choose the observation model, the structure linking variables, and priors ([[random_variables]], [[probabilistic_models]]).
2. Fit: maximise the likelihood $L(\theta)=\prod_ip(y_i\mid\theta)$, or compute the posterior $p(\theta\mid y)\propto p(y\mid\theta)p(\theta)$ ([[parameter_estimation]], [[bayesian_models]]).
3. Check: posterior predictive checks, residuals, calibration, and parameter recovery on simulated data ([[validation_methods]]).
4. Compare: trade fit against complexity ([[information_criteria]], [[cross_validation]]).
5. Report uncertainty on parameters and predictions ([[uncertainty_quantification]]).

Fitting by maximum likelihood minimises $D_{\mathrm{KL}}(\hat p_{\text{data}}\,\|\,p_\theta)$ over the empirical distribution, which ties estimation to [[kl_divergence]].

## Role in Active Inference
The discrete model in `code/tools/src/models/active_inference/generative_model.py` is a small statistical model with five parameter blocks: likelihood $A[o,s]$, transitions $B[s',s,a]$, log preferences $C[o]$, state prior $D[s]$ and action prior $E[a]$. The `DiscreteGenerativeModel` constructor enforces the model's constraints (column-stochastic $A$ and $B$, matching dimensions, finite non-negative entries). The package provides building blocks for the specification step:

- `MatrixInitializer` (`models/matrices/matrix_ops.py`) draws reproducible starting values: `random_stochastic` accepts a `numpy` generator, `identity_based` builds a sticky transition matrix, `uniform` gives an uninformative one.
- `MatrixLoader.validate_matrix` checks a loaded array against shape and constraint specifications read from YAML front matter.
- The package does not implement likelihood fitting, posterior predictive checks or model comparison for fitting agents to data; steps 2 to 4 above are outside its scope.

Statistical modeling of agent behaviour (fitting parameters to choices) is a separate task from the agent's own inference, and conflating them is a common source of confusion: the agent infers states under a fixed model, while the analyst infers the model's parameters from the agent's behaviour.

## Related concepts
- [[statistical_foundations]] — probability and inference background
- [[probabilistic_models]] — model classes and graphical structure
- [[generative_models]] — the agent's own model
- [[model_comparison]] — selecting among fitted models
- [[matrix_operations]] — handling $A$, $B$ as stochastic matrices
- [[computational_methods]] — numerical techniques for fitting
