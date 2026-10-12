---
title: Maximum Entropy Distributions
description: The distributions that maximise entropy under stated constraints, including the uniform, exponential and Gaussian cases, the exponential-family form of the general solution and its derivation by Lagrange multipliers.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - exponential_family
  - probability_distributions
semantic_relations:
  - type: relates
    links:
      - "[[maximum_entropy_principle]]"
      - "[[exponential_family]]"
      - "[[gaussian_distributions]]"
      - "[[differential_entropy]]"
      - "[[gibbs_distributions]]"
---

# Maximum Entropy Distributions

## Overview
Among all probability distributions that satisfy a given set of constraints, the maximum entropy distribution is the one with the greatest Shannon (or differential) entropy. Under the maximum entropy principle of Jaynes (1957) it is the least committal choice consistent with what is known. The principle selects a distribution, and the resulting families coincide with many named distributions.

## General solution
Suppose the constraints are $\mathbb{E}[f_k(X)]=m_k$ for $k=1,\dots,K$, together with normalisation. Introducing Lagrange multipliers $\lambda_k$ and setting the variation of $-\int p\ln p$ to zero gives
$$p(x)=\frac{1}{Z(\lambda)}\exp\!\Bigl(\sum_{k=1}^K\lambda_kf_k(x)\Bigr),\qquad Z(\lambda)=\int\exp\!\Bigl(\sum_k\lambda_kf_k(x)\Bigr)dx,$$
which is an exponential family with $f_k$ as sufficient statistics. The multipliers are fixed by requiring the constraints to hold, equivalently by maximum likelihood in the family. The constrained problem is concave in $p$, so the solution, when it exists, is unique. Entropy is measured relative to a base measure; for continuous variables the answer depends on that choice.

## Standard cases
- **No constraint, finite support:** the uniform distribution.
- **Support $[0,\infty)$ with fixed mean $\mu$:** the exponential distribution with density $\mu^{-1}e^{-x/\mu}$.
- **Support $\mathbb{R}$ with fixed mean and variance:** the Gaussian $\mathcal{N}(\mu,\sigma^2)$, with differential entropy $\tfrac12\ln(2\pi e\sigma^2)$ (Cover and Thomas, 2006).
- **Fixed mean energy over states:** the Gibbs distribution $p\propto e^{-\beta E}$.
- **Fixed mean of $\ln x$ on $[1,\infty)$:** a Pareto-type power-law distribution.

## Relation to Active Inference
The Gaussian result gives a justification for the common assumption of Gaussian beliefs and noise when only mean and variance are specified. Softmax distributions over policies are Gibbs distributions and so have a maximum entropy reading, and Active Inference treats the free energy as the quantity such distributions minimise. These readings are available but are not required by the formal framework.

## Related concepts
- [[maximum_entropy_principle]] — the inference principle that selects these distributions
- [[exponential_family]] — the parametric class that contains all solutions
- [[gibbs_distributions]] — maximum entropy at fixed mean energy
- [[gaussian_distributions]] — maximum entropy for given mean and variance
- [[constraint_optimization]] — the mathematical method used to derive the form
- [[differential_entropy]] — the entropy measure for continuous variables

## References
- Jaynes, E. T. (1957). Information theory and statistical mechanics. Physical Review, 106, 620-630.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
- Wainwright, M. J., & Jordan, M. I. (2008). Graphical models, exponential families, and variational inference. Foundations and Trends in Machine Learning, 1, 1-305.
