---
title: Conditional Independence Tests
description: Statistical procedures that assess whether two variables are independent given a conditioning set, including partial correlation, contingency-table and kernel tests, and their role in learning graphical model structure.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistics
  - causal_discovery
  - graphical_models
semantic_relations:
  - type: relates
    links:
      - "[[conditional_independence]]"
      - "[[bayesian_networks]]"
      - "[[d_separation]]"
      - "[[causal_inference]]"
      - "[[conditional_mutual_information]]"
---

# Conditional Independence Tests

## Overview
A conditional independence (CI) test decides whether the data are compatible with $X\perp\!\!\!\perp Y\mid Z$, meaning $p(x,y\mid z)=p(x\mid z)\,p(y\mid z)$. These tests are the empirical building block of constraint-based structure learning: each accepted or rejected independence statement eliminates edges or fixes orientations in a graph. The PC algorithm of Spirtes, Glymour and Scheines starts from a complete graph and removes the edge between $X$ and $Y$ whenever some conditioning set renders them independent (Spirtes et al., 2000).

## Common tests
- **Partial correlation (Fisher's $z$)**: for jointly Gaussian variables, conditional independence is equivalent to zero partial correlation $\rho_{XY\cdot Z}$. With $n$ samples and $|Z|$ conditioning variables, $\sqrt{n-|Z|-3}\;\tfrac12\ln\frac{1+\hat\rho}{1-\hat\rho}$ is approximately standard normal under the null.
- **Contingency-table tests**: for discrete data, the $\chi^2$ or likelihood-ratio ($G$) statistic is computed within each stratum of $Z$ and summed. The degrees of freedom grow with the number of strata, so power falls quickly as $Z$ grows.
- **Information-theoretic tests**: estimate the conditional mutual information and compare it with a null distribution, often obtained by permuting $X$ within strata of $Z$.
- **Kernel and regression-based tests**: for continuous non-Gaussian data, measures of residual dependence after regressing each variable on $Z$.

## Limitations
Failing to reject the null is not evidence of independence, particularly with small samples. Shah and Peters (2020) showed that, for continuous conditioning variables, no test can have power against all alternatives while controlling the level, so valid tests need assumptions about the data-generating process. Errors in early tests also propagate through the constraint-based search, and repeated testing needs multiple-comparison control.

## Relation to Active Inference
The relation is indirect. Active Inference generative models are probabilistic graphical models whose conditional independence structure defines the factorisation of beliefs, and Markov blanket partitions are statements of conditional independence between internal and external states. CI tests are one way to recover such structure from data, but structure learning in Active Inference more commonly proceeds by comparing models with free energy or Bayesian model reduction. The link is therefore one of shared formalism, not of method.

## Related concepts
- [[conditional_independence]] — the property being tested
- [[d_separation]] — graphical criterion for implied independencies
- [[bayesian_networks]] — models whose structure the tests help recover
- [[causal_inference]] — the main application area of constraint-based discovery
- [[markov_blanket]] — a conditional independence boundary

## References
- Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, Prediction, and Search (2nd ed.). MIT Press.
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference (2nd ed.). Cambridge University Press.
- Shah, R. D., & Peters, J. (2020). The hardness of conditional independence testing and the generalised covariance measure. Annals of Statistics, 48, 1514-1538.
