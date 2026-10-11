---
title: Expectation Maximization
description: The iterative algorithm for maximum likelihood estimation with latent variables, its E and M steps, the free energy view that proves monotone improvement, and its relation to variational Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - inference
  - latent_variables
  - optimisation
semantic_relations:
  - type: relates
    links:
      - "[[maximum_likelihood]]"
      - "[[variational_inference]]"
      - "[[variational_free_energy]]"
      - "[[jensen_inequality]]"
      - "[[parameter_estimation]]"
---

# Expectation Maximization

## Overview
Expectation maximization (EM) is an iterative method for finding maximum likelihood or maximum a posteriori parameter estimates when the data are incomplete or the model contains latent variables. Dempster, Laird and Rubin (1977) gave the general formulation and named it, unifying many earlier special cases. It is widely used for mixture models, hidden Markov models and factor analysis.

## The algorithm
Let $x$ be the observed data, $z$ the latent variables and $\theta$ the parameters. The marginal likelihood $p(x\mid\theta)=\sum_z p(x,z\mid\theta)$ is often hard to maximise directly. EM alternates two steps from a current estimate $\theta^{(t)}$:
- **E step**: compute the posterior over latent variables, $q(z)=p(z\mid x,\theta^{(t)})$, and form the expected complete-data log-likelihood $Q(\theta\mid\theta^{(t)})=\mathbb{E}_{q}[\ln p(x,z\mid\theta)]$.
- **M step**: set $\theta^{(t+1)}=\arg\max_\theta Q(\theta\mid\theta^{(t)})$.

Each iteration does not decrease $\ln p(x\mid\theta)$. The algorithm converges to a stationary point, which can be a local maximum or a saddle, so initialisation matters.

## The free energy view
For any distribution $q(z)$ there is the exact decomposition

$$\ln p(x\mid\theta)=\mathcal{F}(q,\theta)+D_{\mathrm{KL}}\big(q(z)\,\|\,p(z\mid x,\theta)\big),\quad \mathcal{F}(q,\theta)=\mathbb{E}_q\!\left[\ln\frac{p(x,z\mid\theta)}{q(z)}\right].$$

The KL term is non-negative by [[jensen_inequality]], so $\mathcal{F}$ is a lower bound on $\ln p(x\mid\theta)$. Neal and Hinton (1998) showed that EM is coordinate ascent on $\mathcal{F}$: the E step sets $q$ to the exact posterior, closing the KL gap, and the M step maximises $\mathcal{F}$ over $\theta$. This explains monotonic improvement and justifies variants. Partial E steps, which only improve $q$ rather than solving it exactly, still increase the bound, and replacing the exact posterior by a restricted family gives variational EM.

## Relation to Active Inference
The connection is close. The bound $\mathcal{F}$, written with a negative sign, is the variational free energy of Active Inference, so perception (updating $q$ over states) corresponds to an E step and learning (updating parameters) to an M step (see [[variational_free_energy]]). Friston's dynamic expectation maximisation for continuous-time models and the alternating state and parameter updates in discrete-state schemes are explicitly of this form. A difference is that Active Inference usually keeps a distribution over parameters, as in Bayesian learning with Dirichlet counts, rather than a point estimate from the M step. Active Inference also adds action, which has no counterpart in basic EM.

## Related concepts
- [[maximum_likelihood]] — the objective EM maximises
- [[variational_inference]] — generalises the E step to approximate posteriors
- [[variational_free_energy]] — the same functional as the EM lower bound
- [[jensen_inequality]] — proves the bound
- [[parameter_estimation]] — the task EM serves

## References
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society, Series B, 39, 1-38.
- Neal, R. M., & Hinton, G. E. (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. In M. I. Jordan (Ed.), Learning in Graphical Models (pp. 355-368). Kluwer.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
