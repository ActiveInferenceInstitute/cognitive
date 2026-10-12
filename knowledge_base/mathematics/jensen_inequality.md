---
title: Jensen Inequality
description: The inequality relating the expectation of a convex function to the function of the expectation, its proof sketch, and its role in bounding log evidence and establishing non-negative divergences.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - convexity
  - inequalities
  - variational_bounds
semantic_relations:
  - type: relates
    links:
      - "[[kl_divergence]]"
      - "[[variational_bounds]]"
      - "[[variational_free_energy]]"
      - "[[information_gain]]"
      - "[[expectation_maximization]]"
wikidata: Q107203920
---

# Jensen Inequality

## Overview
Jensen's inequality states that for a convex function $\varphi$ and a random variable $X$ with finite expectation,

$$\varphi\big(\mathbb{E}[X]\big)\;\le\;\mathbb{E}\big[\varphi(X)\big].$$

For a concave function the inequality reverses. Equality holds when $\varphi$ is affine on the support of $X$, or when $X$ is almost surely constant, and for strictly convex $\varphi$ only in the constant case. The result was established by Jensen (1906).

## Intuition and proof sketch
A convex function lies above each of its tangent lines. Take the tangent at the point $\mu=\mathbb{E}[X]$, giving $\varphi(x)\ge\varphi(\mu)+c\,(x-\mu)$ for some slope $c$. Taking expectations eliminates the linear term, because $\mathbb{E}[X-\mu]=0$, and leaves $\mathbb{E}[\varphi(X)]\ge\varphi(\mu)$. For a finite weighted sum this is the statement that $\varphi\big(\sum_i w_i x_i\big)\le\sum_i w_i\varphi(x_i)$ for weights $w_i\ge 0$ summing to 1. Familiar consequences include the arithmetic-geometric mean inequality (take $\varphi=-\ln$) and the fact that the variance is non-negative (take $\varphi(x)=x^2$).

## Uses in information theory and inference
- **Non-negativity of KL divergence.** Since $-\ln$ is convex, $D_{\mathrm{KL}}(P\|Q)=\mathbb{E}_P[-\ln(q/p)]\ge-\ln\mathbb{E}_P[q/p]\ge 0$ (see [[kl_divergence]]). The same argument shows that mutual information is non-negative.
- **Bound on log evidence.** For any distribution $q(z)$, concavity of $\ln$ gives $\ln p(x)=\ln\mathbb{E}_q\!\left[\tfrac{p(x,z)}{q(z)}\right]\ge\mathbb{E}_q\!\left[\ln\tfrac{p(x,z)}{q(z)}\right]$, the evidence lower bound.
- **Monotonic improvement of EM.** The same lower bound underlies the proof that each EM step does not decrease the likelihood (see [[expectation_maximization]]).

## Relation to Active Inference
Variational free energy is the negative of this bound, so one direct derivation of the claim that free energy upper-bounds surprise uses Jensen's inequality. That is a mathematical identity rather than a hypothesis: the gap between free energy and surprise is the KL divergence from the approximate to the true posterior, which is non-negative by the argument above. Many information-theoretic terms in expected free energy, such as information gain, are likewise guaranteed non-negative by convexity. The inequality itself carries no commitment about cognition; it is part of the toolkit that makes the variational bounds in [[variational_free_energy]] valid.

## Related concepts
- [[kl_divergence]] — non-negativity follows from the inequality
- [[variational_bounds]] — lower bounds on log evidence
- [[variational_free_energy]] — the negative of the Jensen bound on log evidence
- [[information_gain]] — expected KL divergence, non-negative by convexity
- [[expectation_maximization]] — monotone likelihood increase proved via the bound

## References
- Jensen, J. L. W. V. (1906). Sur les fonctions convexes et les inegalites entre les valeurs moyennes. Acta Mathematica, 30, 175-193.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
- Boyd, S., & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.
