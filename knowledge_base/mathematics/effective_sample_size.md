---
title: Effective Sample Size
description: The number of independent, equally weighted samples that would give the same estimator variance as a set of weighted or autocorrelated Monte Carlo draws, with formulas for importance sampling and MCMC.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - monte_carlo
  - sampling
  - diagnostics
semantic_relations:
  - type: relates
    links:
      - "[[importance_sampling]]"
      - "[[markov_chain_monte_carlo]]"
      - "[[monte_carlo_methods]]"
      - "[[particle_filtering]]"
      - "[[sample_complexity]]"
---

# Effective Sample Size

## Overview
Monte Carlo estimates improve as the number of draws grows, but draws that are weighted unequally or correlated carry less information than independent ones. The effective sample size (ESS) is the number $N_{\text{eff}}$ of independent, equally weighted draws that would give an estimator the same variance. It is a diagnostic for the quality of an approximation, and for importance weights it is at most the nominal sample size $N$. For correlated MCMC draws it is usually smaller than $N$, but negatively autocorrelated chains can give a value above $N$.

## Importance sampling
With unnormalised weights $w_i$ for $N$ draws, the Kish approximation is
$$N_{\text{eff}}=\frac{\left(\sum_{i=1}^N w_i\right)^2}{\sum_{i=1}^N w_i^2}=\frac{1}{\sum_i \tilde w_i^2},$$
where $\tilde w_i=w_i/\sum_j w_j$ are the normalised weights. It equals $N$ when all weights are equal and approaches $1$ when one weight dominates. It is a heuristic derived from a variance approximation for a self-normalised estimator, not an exact identity for every integrand.

## Markov chain Monte Carlo
For a stationary chain with autocorrelations $\rho_k$ at lag $k$, the variance of the sample mean is approximately $\sigma^2/N_{\text{eff}}$ with
$$N_{\text{eff}}=\frac{N}{1+2\sum_{k=1}^{\infty}\rho_k}.$$
The denominator is the integrated autocorrelation time. In practice the sum is truncated using estimates from the chain, and several chains are combined to guard against failure to explore all modes. A low ESS indicates slow mixing, and reporting it alongside the potential scale reduction statistic is standard practice.

## Relation to Active Inference
Implementations that approximate beliefs by samples need this diagnostic. In particle filtering a drop in $N_{\text{eff}}$ signals weight degeneracy and is the usual trigger for resampling. Sampling-based accounts of neural inference use the same ideas, though most Active Inference schemes use variational rather than sampling approximations, so the connection is an implementation one and not part of the theory.

## Related concepts
- [[importance_sampling]] — weighted draws, where the Kish formula applies
- [[markov_chain_monte_carlo]] — correlated draws, where autocorrelation reduces ESS
- [[particle_filtering]] — resampling decisions based on ESS
- [[hamiltonian_monte_carlo]] — a sampler designed to reduce autocorrelation
- [[metropolis_hastings]] — a basic sampler whose chains are often highly autocorrelated

## References
- Kish, L. (1965). Survey Sampling. Wiley.
- Liu, J. S. (2001). Monte Carlo Strategies in Scientific Computing. Springer.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.
