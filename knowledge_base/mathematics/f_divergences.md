---
title: F-Divergences
description: The Ali-Silvey-Csiszár family of divergences defined by a convex generator function, with KL, total variation, chi-squared and Jensen-Shannon as special cases, and their use in variational inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - divergences
  - information_theory
  - variational_inference
semantic_relations:
  - type: relates
    links:
      - "[[kl_divergence]]"
      - "[[jensen_shannon_divergence]]"
      - "[[jensen_inequality]]"
      - "[[information_theory]]"
      - "[[variational_inference]]"
---

# F-Divergences

## Overview
An f-divergence measures how different two probability distributions are by averaging a convex function of their likelihood ratio. Given a convex function $f:(0,\infty)\to\mathbb{R}$ with $f(1)=0$, and distributions $P$ and $Q$ with densities $p$ and $q$ (with $P$ absolutely continuous with respect to $Q$),

$$D_f(P\,\|\,Q)=\int q(x)\,f\!\left(\frac{p(x)}{q(x)}\right)dx.$$

The family was introduced by Ali and Silvey (1966) and, independently, by Csiszár (1967).

## Properties
- **Non-negativity.** By [[jensen_inequality]], $D_f(P\|Q)\ge f(1)=0$, with equality when $P=Q$ if $f$ is strictly convex at 1.
- **Data processing.** Applying the same stochastic map to both distributions cannot increase $D_f$. This is the property that makes the family suitable for comparing statistical experiments.
- **Joint convexity** in the pair $(P,Q)$.
- **Not symmetric or metric in general.** Swapping arguments corresponds to the generator $f^{*}(t)=t\,f(1/t)$, and the triangle inequality usually fails.

## Familiar members
- $f(t)=t\ln t$ gives the Kullback-Leibler divergence, $D_{\mathrm{KL}}(P\|Q)$ (see [[kl_divergence]]).
- $f(t)=-\ln t$ gives the reverse KL divergence.
- $f(t)=\tfrac12|t-1|$ gives total variation distance.
- $f(t)=(t-1)^2$ gives the chi-squared divergence.
- $f(t)=(\sqrt t-1)^2$ gives the squared Hellinger distance, up to a normalising constant.
- $f(t)=\tfrac12\big[t\ln t-(t+1)\ln\tfrac{t+1}{2}\big]$ gives the [[jensen_shannon_divergence]], the average KL divergence of $P$ and $Q$ to their mixture.

## Relation to Active Inference
Standard variational free energy minimises the KL divergence between an approximate posterior and the true posterior, so Active Inference is built on one member of the family. Work on variational inference more broadly has explored other members, including alpha-divergences and the chi-squared divergence, which change how the approximation trades mass-covering against mode-seeking behaviour; reversing the KL direction, for example, yields the expectation-propagation style of approximation. Applying such alternatives inside Active Inference is an area of active exploration, not part of the standard formulation. The family is mainly useful here as a unifying vocabulary for comparing beliefs, and as a source of bounds in [[variational_inference]].

## Related concepts
- [[kl_divergence]] — the member used by variational free energy
- [[jensen_shannon_divergence]] — symmetric, bounded relative of KL
- [[jensen_inequality]] — gives non-negativity from convexity
- [[information_theory]] — the broader setting for divergence measures
- [[information_geometry]] — locally, f-divergences induce the Fisher metric

## References
- Ali, S. M., & Silvey, S. D. (1966). A general class of coefficients of divergence of one distribution from another. Journal of the Royal Statistical Society, Series B, 28, 131-142.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
