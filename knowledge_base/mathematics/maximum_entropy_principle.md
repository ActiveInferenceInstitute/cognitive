---
title: Maximum Entropy Principle
description: The rule of choosing, among distributions consistent with known constraints, the one with greatest entropy, its exponential-family solution, standard examples, and its place in statistical physics and Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - statistical_physics
  - inference
semantic_relations:
  - type: relates
    links:
      - "[[entropy]]"
      - "[[differential_entropy]]"
      - "[[exponential_family]]"
      - "[[information_projection]]"
      - "[[statistical_physics]]"
      - "[[gaussian_distributions]]"
---

# Maximum Entropy Principle

## Overview
The maximum entropy principle, formulated by E. T. Jaynes, says that when only partial information about a distribution is available, expressed as constraints, the least biased choice is the distribution of maximum entropy that satisfies them. It is a rule for assigning probabilities, not an empirical claim, and it commits the modeller to no structure beyond the stated constraints.

## Solution
Suppose the constraints are $\mathbb{E}[f_k(x)]=c_k$ for $k=1,\dots,K$. Maximising the entropy $H[p]=-\int p\ln p$ subject to these and to normalisation, using Lagrange multipliers, gives
$$p(x)=\frac{1}{Z(\lambda)}\exp\Big(-\sum_k\lambda_k f_k(x)\Big),$$
where the multipliers $\lambda_k$ are fixed by the constraints and $Z$ is the partition function. The solution is therefore an exponential family distribution.

## Standard cases
- A finite set of outcomes with no constraint: the uniform distribution.
- A positive variable with a given mean: the exponential distribution.
- A real variable with given mean and variance: the Gaussian distribution.
- A system with fixed mean energy: the Boltzmann distribution, which links the principle to equilibrium statistical mechanics.

The differential entropy used for continuous variables depends on the coordinates unless a reference measure is specified, which is why the principle is often stated in terms of relative entropy. Minimising the Kullback-Leibler divergence from a prior subject to constraints generalises it and gives the information projection.

## Relation to Active Inference
The principle appears in several places in the framework. The Gaussian form of beliefs under the Laplace assumption is the maximum entropy choice given a mean and covariance. Variational free energy itself contains an entropy term, so minimising it favours approximate posteriors that are as broad as the data allow, and the expected free energy likewise rewards policies that retain options. The steady-state densities used in the free energy principle are often written in exponential form, which is the maximum entropy form for a given set of constraints. These are uses of the principle as a modelling default, and the choice of constraints remains a judgement of the modeller.

## Related concepts
- [[entropy]] — the quantity being maximised
- [[exponential_family]] — the form every maximum entropy solution takes
- [[information_projection]] — the relative-entropy generalisation
- [[statistical_physics]] — origin of the Boltzmann and Gibbs distributions
- [[differential_entropy]] — the continuous-variable analogue
- [[epistemic_value]] — information-seeking term in Active Inference

## References
- Jaynes, E. T. (1957). Information theory and statistical mechanics. Physical Review, 106, 620-630.
- Jaynes, E. T. (1957). Information theory and statistical mechanics. II. Physical Review, 108, 171-190.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
