---
title: Bayesian Optimization
description: Sample-efficient global optimisation of expensive black-box functions using a probabilistic surrogate and an acquisition function, and its structural parallels with epistemic value in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimization
  - surrogate_models
  - exploration
semantic_relations:
  - type: relates
    links:
      - "[[gaussian_processes]]"
      - "[[epistemic_value]]"
      - "[[exploration_exploitation]]"
      - "[[hyperparameter_optimization]]"
      - "[[information_gain]]"
      - "[[annealing_schedules]]"
---

# Bayesian Optimization

## Overview
Bayesian optimization is a strategy for finding the maximiser $x^*=\arg\max_{x\in\mathcal{X}} f(x)$ of a function that is expensive to evaluate, has no usable gradient and may be noisy. Instead of probing $f$ densely, it maintains a probabilistic surrogate model of $f$ and uses the model's uncertainty to decide which point to evaluate next. It is used where each evaluation costs a lengthy experiment or training run, as in [[hyperparameter_optimization]].

## Procedure
The method repeats a short loop.
1. Fit a surrogate to the observations so far. The usual choice is a [[gaussian_processes|Gaussian process]], which gives a posterior mean $\mu(x)$ and standard deviation $\sigma(x)$ at every candidate point.
2. Maximise an acquisition function $\alpha(x)$ computed from the posterior. This inner problem is cheap because it uses only the surrogate.
3. Evaluate $f$ at the chosen point, add the result to the data and return to step 1.

Common acquisition functions include probability of improvement, expected improvement, the upper confidence bound $\mu(x)+\kappa\,\sigma(x)$, Thompson sampling, and information-theoretic criteria such as entropy search. Each encodes a balance between exploiting regions with high predicted value and exploring regions of high uncertainty (see [[exploration_exploitation]]). The parameter $\kappa$ in the upper confidence bound sets this balance directly.

## Relation to Active Inference
The link is a structural analogy. Expected free energy separates into a pragmatic term, which favours outcomes consistent with preferences, and an epistemic term, which favours actions expected to reduce uncertainty about hidden states or parameters (see [[epistemic_value]]). Acquisition functions play a similar role: the mean term corresponds to exploitation and the uncertainty bonus to information seeking. Entropy-search criteria are closely related to expected [[information_gain]]. The two are not equivalent in general. Bayesian optimization targets the location of an optimum of an objective, whereas expected free energy is defined over a generative model of observations with prior preferences.

## Related concepts
- [[gaussian_processes]] — the usual surrogate model
- [[epistemic_value]] — the uncertainty-reducing term in expected free energy
- [[exploration_exploitation]] — the trade-off that acquisition functions resolve
- [[hyperparameter_optimization]] — a leading application
- [[information_gain]] — the quantity targeted by entropy-based acquisition

## References
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. Journal of Global Optimization, 13, 455-492.
- Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the human out of the loop: A review of Bayesian optimization. Proceedings of the IEEE, 104, 148-175.
- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.
