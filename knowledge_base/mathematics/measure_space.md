---
title: Measure Space
description: A set equipped with a sigma-algebra and a measure, the basic structure of integration and probability theory, including probability spaces, densities, and the difficulties of infinite-dimensional spaces of paths.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - measure_theory
  - probability_theory
  - integration
semantic_relations:
  - type: relates
    links:
      - "[[measure_theory]]"
      - "[[probability_theory]]"
      - "[[random_variables]]"
      - "[[kolmogorov_extension]]"
      - "[[integration_theory]]"
wikidata: Q3058212
---

# Measure Space

## Overview
A measure space is the setting in which size, integration and probability are made precise. It consists of a set $\Omega$, a collection of subsets $\mathcal{F}$ called a $\sigma$-algebra that specifies which sets can be measured, and a measure $\mu$ that assigns each such set a non-negative size. Probability theory is the special case in which the total size is one.

## Definition
A $\sigma$-algebra $\mathcal{F}$ on $\Omega$ contains $\Omega$, is closed under complements and is closed under countable unions. A measure $\mu:\mathcal{F}\to[0,\infty]$ satisfies $\mu(\varnothing)=0$ and countable additivity,
$$\mu\Big(\bigcup_i A_i\Big)=\sum_i\mu(A_i)$$
for pairwise disjoint sets $A_i$. The triple $(\Omega,\mathcal{F},\mu)$ is a measure space, and if $\mu(\Omega)=1$ it is a probability space. Examples are counting measure on a finite set, Lebesgue measure on $\mathbb{R}^n$, and the distribution of a random variable on its range.

## Key notions
- **Integration**: the Lebesgue integral $\int f\,d\mu$ is defined for measurable functions, and expectations are integrals against a probability measure.
- **Null sets**: sets of measure zero are negligible, and statements that hold outside them are said to hold almost everywhere.
- **Densities**: if $\nu$ is absolutely continuous with respect to $\mu$, the Radon-Nikodym theorem gives a density $d\nu/d\mu$. The Kullback-Leibler divergence is defined through such a density, $D_{\mathrm{KL}}(\nu\|\mu)=\int \ln\frac{d\nu}{d\mu}\,d\nu$.
- **Product spaces**: the product of measure spaces is again one, which underlies independence and joint distributions.

## Infinite-dimensional spaces
Spaces of paths or fields are infinite-dimensional, and no analogue of Lebesgue measure exists there that is invariant under translations. Measures on such spaces, for example the Wiener measure on continuous paths, must be constructed indirectly (see [[kolmogorov_extension]] and [[wiener_process]]). This is why path integrals require care in a rigorous treatment.

## Relation to Active Inference
Measure theory is the foundation for stating probability densities, divergences and conditional distributions precisely. Standard Active Inference works with finite or Euclidean spaces where elementary densities suffice. Measure-theoretic treatments are used to make the free energy principle rigorous on general spaces, and to handle path-space formulations.

## Related concepts
- [[measure_theory]] — the broader theory of measures and integration
- [[probability_theory]] — measure spaces with total mass one
- [[kolmogorov_extension]] — construction of measures on path space
- [[kl_divergence]] — defined through Radon-Nikodym derivatives
- [[measure_theoretic_active_inference]] — measure-theoretic treatment of the formalism
- [[path_integral_theory]] — integrals over spaces of paths

## References
- Folland, G. B. (1999). Real Analysis: Modern Techniques and Their Applications (2nd ed.). Wiley.
- Billingsley, P. (1995). Probability and Measure (3rd ed.). Wiley.
