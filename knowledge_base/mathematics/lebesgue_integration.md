---
title: Lebesgue Integration
description: Integration with respect to a measure, defined through simple functions, with its convergence theorems and the reasons it underlies expectation, L^p spaces and integration on function spaces.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - measure_theory
  - real_analysis
  - probability
semantic_relations:
  - type: relates
    links:
      - "[[measure_theory]]"
      - "[[integration_theory]]"
      - "[[real_analysis]]"
      - "[[probability_theory]]"
      - "[[functional_analysis]]"
wikidata: Q853831
---

# Lebesgue Integration

## Overview
Lebesgue integration defines the integral of a function with respect to a measure $\mu$ on a measurable space. Instead of partitioning the domain into intervals, as the Riemann integral does, it partitions the range and measures the size of the sets where the function takes values in each part. This enlarges the class of integrable functions and gives much better behaviour under limits.

## Construction
For a simple function $s=\sum_{i=1}^k a_i\mathbf{1}_{A_i}$ with measurable $A_i$, set $\int s\,d\mu=\sum_i a_i\,\mu(A_i)$. For a measurable $f\ge0$, define
$$\int f\,d\mu=\sup\left\{\int s\,d\mu:\ 0\le s\le f,\ s\text{ simple}\right\}.$$
A general $f$ is integrable if $\int|f|\,d\mu<\infty$, and then $\int f\,d\mu=\int f^+d\mu-\int f^-d\mu$. The indicator of the rationals has Lebesgue integral zero, since the rationals have measure zero, but it is not Riemann integrable. A bounded function on a compact interval is Riemann integrable exactly when it is continuous almost everywhere, and in that case the two integrals agree.

## Convergence theorems
- **Monotone convergence.** If $0\le f_n\uparrow f$ pointwise, then $\int f_n\,d\mu\uparrow\int f\,d\mu$.
- **Fatou's lemma.** For $f_n\ge0$, $\int\liminf f_n\,d\mu\le\liminf\int f_n\,d\mu$.
- **Dominated convergence.** If $f_n\to f$ almost everywhere and $|f_n|\le g$ with $g$ integrable, then $\int f_n\,d\mu\to\int f\,d\mu$.

These allow limits and integrals to be exchanged under mild conditions. They also make the spaces $L^p(\mu)$ complete (Riesz-Fischer theorem), which is why $L^2$ is a Hilbert space.

## Relation to Active Inference
In probability, the expectation of a random variable is its Lebesgue integral against the probability measure, $\mathbb{E}[X]=\int X\,dP$, and the Kullback-Leibler divergence is defined by integrating a Radon-Nikodym derivative. Free energies and expected free energies are expectations of this kind. The framework is also needed in the path integral setting: there is no Lebesgue measure on infinite-dimensional path space, so integrals over paths are defined with respect to measures such as Wiener measure rather than by naive limits. These uses are foundational and standard.

## Related concepts
- [[measure_theory]] — the construction of measures on which the integral depends
- [[measure_space]] — the triple $(X,\mathcal{B},\mu)$ over which integration is defined
- [[integration_theory]] — the broader theory of integrals
- [[radon_nikodym_theorem]] — densities of one measure with respect to another
- [[gaussian_measures]] — measures on path space used in stochastic integrals

## References
- Rudin, W. (1987). Real and Complex Analysis (3rd ed.). McGraw-Hill.
- Folland, G. B. (1999). Real Analysis: Modern Techniques and Their Applications (2nd ed.). Wiley.
- Royden, H. L., & Fitzpatrick, P. M. (2010). Real Analysis (4th ed.). Pearson.
