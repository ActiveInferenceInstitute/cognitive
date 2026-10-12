---
title: Adaptive MCMC
description: Markov chain Monte Carlo methods that tune their proposal distribution or step size from the chain's own history, with the adaptive Metropolis algorithm, convergence conditions and relevance to Bayesian inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - sampling
  - bayesian_computation
  - markov_chains
semantic_relations:
  - type: relates
    links:
      - "[[markov_chain_monte_carlo]]"
      - "[[metropolis_hastings]]"
      - "[[hamiltonian_monte_carlo]]"
      - "[[dual_averaging]]"
      - "[[bayesian_inference]]"
---

# Adaptive MCMC

## Overview
Adaptive Markov chain Monte Carlo refers to algorithms that adjust their own transition kernel while running, using information gathered from earlier samples. The motivation is that the efficiency of a sampler such as random-walk Metropolis depends strongly on the proposal scale and shape, which are rarely known in advance for a posterior in many dimensions. Rather than tuning by hand over pilot runs, adaptive methods learn suitable settings on the fly.

## Adaptive Metropolis
In random-walk Metropolis, a candidate $y\sim\mathcal N(x_t,\Sigma)$ is accepted with probability $\min\{1,\pi(y)/\pi(x_t)\}$. For a target close to Gaussian in $d$ dimensions, theory of optimal scaling suggests proposal covariance of about $(2.38^2/d)\,\Sigma_\pi$, where $\Sigma_\pi$ is the target covariance, with an acceptance rate near 0.234 in high dimensions (Roberts, Gelman and Gilks, 1997). Since $\Sigma_\pi$ is unknown, the adaptive Metropolis algorithm of Haario, Saksman and Tamminen (2001) estimates it by the empirical covariance of past samples,
$$\Sigma_t = s_d\,\mathrm{Cov}(X_0,\dots,X_{t-1}) + s_d\,\epsilon I_d,\qquad s_d\approx \frac{2.4^2}{d},$$
where a small $\epsilon>0$ prevents degeneracy. Other adaptations tune the step size toward a target acceptance rate. In Hamiltonian Monte Carlo, the step size is commonly adapted during warm-up using dual averaging to reach a target acceptance statistic, as in the No-U-Turn Sampler (Hoffman and Gelman, 2014).

## Validity
Adaptation makes the process non-Markovian, so the chain no longer has the target as stationary distribution for every kernel in the sequence, and naive adaptation can produce a chain that converges to the wrong distribution. Roberts and Rosenthal (2007) give sufficient conditions, notably diminishing adaptation, meaning that the change in the kernel between iterations shrinks to zero, together with a containment condition that controls convergence times. The covariance adaptation above satisfies diminishing adaptation because each new sample changes the running estimate by an amount of order $1/t$. A simple safe practice is to adapt only during a warm-up phase and discard those samples, then run a fixed kernel for the final sample.

## Relation to Active Inference
Sampling is an alternative to variational inference, which is the approximation normally used in Active Inference, so the relation is indirect. Adaptive samplers are relevant as reference methods for checking variational posteriors, for path-integral and stochastic control implementations that sample trajectories, and for fitting generative models to data in empirical work. They are not part of the free energy principle itself. See [[variational_inference]].

## Related concepts
- [[markov_chain_monte_carlo]] — the general class of samplers
- [[metropolis_hastings]] — the accept-reject algorithm that is adapted
- [[hamiltonian_monte_carlo]] — a gradient-based sampler with adaptive step sizes
- [[dual_averaging]] — a method for step-size adaptation
- [[bayesian_inference]] — the setting in which posterior sampling is used
- [[variational_inference]] — the optimisation-based alternative

## References
- Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. Annals of Applied Probability, 7, 110-120.
- Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis algorithm. Bernoulli, 7, 223-242.
- Roberts, G. O., & Rosenthal, J. S. (2007). Coupling and ergodicity of adaptive Markov chain Monte Carlo algorithms. Journal of Applied Probability, 44, 458-475.
- Andrieu, C., & Thoms, J. (2008). A tutorial on adaptive MCMC. Statistics and Computing, 18, 343-373.
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15, 1593-1623.
