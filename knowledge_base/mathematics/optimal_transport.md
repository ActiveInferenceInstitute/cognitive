---
title: Optimal Transport
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - optimal_transport
  - wasserstein
  - probability
aliases:
  - wasserstein_metrics
  - wasserstein_geometry
  - transportation_theory
  - wasserstein_metric
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/kl_divergence]]"
      - "[[knowledge_base/mathematics/information_geometry]]"
      - "[[knowledge_base/mathematics/measure_theory]]"
      - "[[knowledge_base/mathematics/probability_theory]]"
---

# Optimal Transport

## Overview
Optimal transport (OT) asks for the cheapest way to move one probability distribution onto another, given a cost of moving unit mass between points. The minimum cost defines distances between distributions, the Wasserstein distances, which respect the geometry of the underlying space. Unlike the KL divergence, they stay finite and informative when the two distributions have disjoint supports.

## Definition / Formalism
For distributions $\mu,\nu$ on a metric space with cost $c(x,y)$, the Kantorovich problem is

$$\min_{\pi\in\Pi(\mu,\nu)}\int c(x,y)\,d\pi(x,y),$$

where $\Pi(\mu,\nu)$ is the set of couplings with marginals $\mu$ and $\nu$. With $c=d^p$ the $p$-Wasserstein distance is $W_p=(\min_\pi\int d(x,y)^p\,d\pi)^{1/p}$.

Closed forms worth knowing:

- On the real line, $W_1=\int|F_\mu(x)-F_\nu(x)|\,dx$ and $W_p$ is computed by matching quantiles. For two point masses at distance $a$, $W_p=a$ while the KL divergence is infinite.
- Two 1-D Gaussians satisfy $W_2^2=(\mu_1-\mu_2)^2+(\sigma_1-\sigma_2)^2$. For $\mathcal N(0,1)$ and $\mathcal N(3,2^2)$ this gives $\sqrt{10}\approx3.162$, and quantile matching on $2\times10^5$ samples returned 3.163.
- Four equally spaced grid points with mass $(\tfrac12,\tfrac12,0,0)$ versus $(0,0,\tfrac12,\tfrac12)$ give $W_1=2$ grid units.

Entropic regularisation adds $-\varepsilon H(\pi)$ to the objective and is solved by Sinkhorn iterations that alternately rescale rows and columns of $K=e^{-C/\varepsilon}$ to match the marginals. On a $2\times2$ example (marginals $(0.7,0.3)$ and $(0.3,0.7)$, cost $1-\delta_{ij}$, $\varepsilon=0.05$) the iteration reproduces the exact transport cost of 0.4.

## Role in Active Inference
Indirect. Standard active inference measures belief change with KL divergence, so OT is not part of the formalism or of this repo's code. Where OT could matter is as a different notion of closeness between beliefs or preference distributions: for ordered or metric outcome spaces, KL treats neighbouring and distant states identically, while a Wasserstein distance does not. Any such use is an extension, not an established part of the framework.

## Related concepts
- [[kl_divergence]] — the divergence OT is usually contrasted with
- [[information_geometry]] — Fisher-Rao geometry versus Wasserstein geometry on distributions
- [[measure_theory]] — couplings and push-forward measures
- [[probability_distributions]] — the objects being transported
- [[statistical_manifolds]] — the manifold OT offers an alternative metric on

## References
- Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
- Peyre, G. and Cuturi, M. (2019). *Computational Optimal Transport*. Foundations and Trends in Machine Learning.
