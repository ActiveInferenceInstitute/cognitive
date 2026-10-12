---
title: Differential Entropy
description: The continuous analogue of Shannon entropy for densities, its closed forms and differences from discrete entropy, and its appearance in Gaussian approximations used in the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - entropy
  - continuous_distributions
semantic_relations:
  - type: relates
    links:
      - "[[entropy]]"
      - "[[information_theory]]"
      - "[[kl_divergence]]"
      - "[[mutual_information]]"
      - "[[free_energy_principle]]"
wikidata: Q3055345
---

# Differential Entropy

## Overview
Differential entropy extends Shannon [[entropy]] to a continuous random variable $X$ with probability density $p(x)$:

$$h(X)=-\int p(x)\,\ln p(x)\,dx.$$

It is measured in nats with the natural logarithm and in bits with base 2. It is a useful summary of how spread out a density is, but it does not inherit every property of the discrete quantity.

## Properties and differences from discrete entropy
- **It can be negative.** A density may exceed 1 where it is concentrated, so $h(X)<0$ is possible. Uniform on $[0,a]$ has $h=\ln a$, which is negative for $a<1$.
- **It is not invariant under change of variables.** For $Y=aX$, $h(Y)=h(X)+\ln|a|$; it depends on the units of measurement. Discretising $X$ with bin width $\Delta$ gives a discrete entropy of about $h(X)-\ln\Delta$, which diverges as $\Delta\to 0$.
- **Relative quantities are well behaved.** [[kl_divergence]] and [[mutual_information]] remain non-negative and invariant under invertible reparameterisation, and mutual information can be written as $I(X;Y)=h(X)+h(Y)-h(X,Y)$.

Standard closed forms include $\tfrac12\ln(2\pi e\,\sigma^2)$ for a univariate Gaussian and $\tfrac12\ln\!\big((2\pi e)^k|\Sigma|\big)$ for a $k$-dimensional Gaussian with covariance $\Sigma$. Among all densities with a given variance, the Gaussian has the largest differential entropy, which is why it is the least committal choice when only the second moment is fixed.

## Relation to Active Inference
Differential entropy is used directly wherever beliefs are continuous. Variational free energy contains the entropy of the approximate posterior as a term that rewards keeping beliefs broad. Under the Laplace approximation the posterior is Gaussian, so that term has the closed form above and depends on the log-determinant of the posterior covariance (see [[variational_free_energy]]). In the free energy principle, the entropy of sensory states is the long-run average of surprise, and it is this quantity that adaptive systems are argued to keep bounded. The principle's treatment of continuous states relies on densities, so the caveats above (units and sign) apply when numerical values are compared across models.

## Related concepts
- [[entropy]] — the discrete quantity this generalises
- [[kl_divergence]] — the invariant relative form used to compare densities
- [[mutual_information]] — expressed through differential entropies
- [[variational_free_energy]] — contains the entropy of the approximate posterior
- [[information_theory]] — the wider framework

## References
- Shannon, C. E. (1948). A mathematical theory of communication. Bell System Technical Journal, 27, 379-423, 623-656.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
