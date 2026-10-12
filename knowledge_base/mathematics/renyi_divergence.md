---
title: Renyi Divergence
description: A one-parameter family of divergences between probability distributions that contains the Kullback-Leibler divergence as a limit, with its definition, special cases, properties and use in variational bounds.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - divergences
  - variational_bounds
semantic_relations:
  - type: relates
    links:
      - "[[kl_divergence]]"
      - "[[f_divergences]]"
      - "[[entropy]]"
      - "[[data_processing_inequality]]"
      - "[[variational_bounds]]"
---

# Renyi Divergence

## Overview
The Renyi divergence of order $\alpha$ generalises the Kullback-Leibler divergence. For discrete distributions $P$ and $Q$ on the same set, and $\alpha>0$ with $\alpha\neq1$,
$$D_\alpha(P\|Q)=\frac{1}{\alpha-1}\ln\sum_xp(x)^\alpha q(x)^{1-\alpha}.$$
For general measures, the sum becomes an integral of $(dP/dQ)^\alpha$ against $Q$. It was introduced by Renyi (1961) alongside the Renyi entropy $H_\alpha(P)=\frac{1}{1-\alpha}\ln\sum_xp(x)^\alpha$, which it parallels in the way that the KL divergence parallels Shannon entropy.

## Special cases and properties
- **Limit $\alpha\to1$.** $D_\alpha$ converges to the Kullback-Leibler divergence $D_{\mathrm{KL}}(P\|Q)$, so $D_1$ is defined by continuity ([[kl_divergence]]).
- **Order $1/2$.** $D_{1/2}(P\|Q)=-2\ln\sum_x\sqrt{p(x)q(x)}$, a function of the Bhattacharyya coefficient and symmetric in $P$ and $Q$.
- **Order $2$.** $D_2(P\|Q)=\ln\big(1+\chi^2(P\|Q)\big)$, where $\chi^2$ is the chi-squared divergence.
- **Order $\infty$.** $D_\infty(P\|Q)=\ln\sup_xp(x)/q(x)$, the logarithm of the largest likelihood ratio.
- **General properties.** $D_\alpha(P\|Q)\ge0$ with equality exactly when $P=Q$. It is non-decreasing in $\alpha$, and it satisfies the data processing inequality: applying the same channel to both distributions cannot increase it ([[data_processing_inequality]]). Unlike a metric, it is not symmetric in general and does not satisfy the triangle inequality. The monotonicity in $\alpha$ and the extension to $\alpha\in[0,\infty]$ are treated carefully by van Erven and Harremoes (2014).

Renyi divergences are related to the $f$-divergences through monotone transformations of integrals of the form $\int (dP/dQ)^\alpha\,dQ$, which are Hellinger-type or chi-type divergences for particular $\alpha$ ([[f_divergences]]).

## Relation to Active Inference
Variational free energy is a bound built from the KL divergence between an approximate posterior $q$ and the true posterior, and variational inference minimises $D_{\mathrm{KL}}(q\|p)$. Replacing this by an $\alpha$-divergence gives alpha-divergence and Renyi variational bounds, which interpolate between the usual evidence lower bound and bounds that favour mass-covering approximations. These are developed in the machine-learning literature on variational inference ([[variational_bounds]], [[variational_inference]]). The standard Active Inference treatment uses the KL form, so the Renyi generalisation is an extension that remains a research direction in the context of the [[free_energy_principle]], not part of its core formalism.

## Related concepts
- [[kl_divergence]] — the limiting case at order one
- [[f_divergences]] — the wider family containing related divergences
- [[entropy]] — the quantity whose Renyi form is parallel
- [[data_processing_inequality]] — a property shared with the divergence
- [[variational_bounds]] — bounds generalised by alpha-divergences
- [[jensen_shannon_divergence]] — another symmetric divergence

## References
- Renyi, A. (1961). On measures of entropy and information. Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability, 1, 547-561.
- van Erven, T., & Harremoes, P. (2014). Renyi divergence and Kullback-Leibler divergence. IEEE Transactions on Information Theory, 60, 3797-3820.
