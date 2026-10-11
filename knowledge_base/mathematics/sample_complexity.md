---
title: Sample Complexity
description: The number of observations needed to learn or estimate a quantity to a stated accuracy with high probability, PAC bounds for finite classes, effects of priors, and implications for active data selection.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - learning_theory
  - statistics
  - generalization
semantic_relations:
  - type: relates
    links:
      - "[[learning_theory]]"
      - "[[active_learning]]"
      - "[[computational_complexity]]"
      - "[[optimal_experiment_design]]"
      - "[[epistemic_value]]"
      - "[[generalization]]"
---

# Sample Complexity

## Overview
The sample complexity of a learning problem is the number of examples required for a learner to reach a given accuracy with a given confidence. In the probably approximately correct (PAC) framework introduced by Valiant, a learner must output, with probability at least $1-\delta$, a hypothesis whose error is at most $\varepsilon$, and the sample complexity is the smallest sample size $m(\varepsilon,\delta)$ for which this is possible. It is a statistical resource, separate from the computation time the learner needs.

## Basic bounds
For a finite hypothesis class $\mathcal{H}$ in the realisable case, where some hypothesis has zero error, any consistent learner succeeds if
$$m\ge\frac{1}{\varepsilon}\Big(\ln|\mathcal{H}|+\ln\frac{1}{\delta}\Big).$$
In the agnostic case, where no hypothesis is perfect, uniform convergence by Hoeffding's inequality and a union bound requires
$$m\ge\frac{1}{2\varepsilon^2}\Big(\ln|\mathcal{H}|+\ln\frac{2}{\delta}\Big)$$
samples to estimate the error of every hypothesis within $\varepsilon$. For infinite classes, the Vapnik-Chervonenkis dimension $d$ replaces $\ln|\mathcal{H}|$ and controls the scaling. Estimating the mean of a bounded variable to accuracy $\varepsilon$ with confidence $1-\delta$ needs on the order of $\varepsilon^{-2}\ln(1/\delta)$ samples. The quadratic dependence on $1/\varepsilon$ explains why precise estimation is expensive.

## Reducing sample needs
Sample complexity falls when the hypothesis class is small, when strong inductive bias or informative priors restrict the candidates, and when the learner chooses which examples to see. Active learning can in some problems give exponential savings over passive sampling, although not in all.

## Relation to Active Inference
The relevance is conceptual. Informative priors and hierarchical generative models, which Bayesian accounts of the brain use, are mechanisms that can reduce the data needed to infer latent causes. In Active Inference, the epistemic term of expected free energy directs the agent toward observations that are expected to reduce uncertainty about hidden states and parameters, which is a form of active data selection in the spirit of optimal experiment design. The framework does not in general supply formal PAC-style sample complexity guarantees, and claims about learning efficiency are normally supported by simulation.

## Related concepts
- [[learning_theory]] — the theory in which the quantity is defined
- [[active_learning]] — choosing informative examples to reduce the sample requirement
- [[optimal_experiment_design]] — formal criteria for selecting measurements
- [[epistemic_value]] — the information-seeking component of Active Inference
- [[generalization]] — performance on unseen data, which the bounds control

## References
- Valiant, L. G. (1984). A theory of the learnable. Communications of the ACM, 27, 1134-1142.
- Vapnik, V. N. (1998). Statistical Learning Theory. Wiley.
- Shalev-Shwartz, S., & Ben-David, S. (2014). Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press.
