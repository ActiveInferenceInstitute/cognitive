---
title: Markov Chain Monte Carlo
description: Sampling methods that build a Markov chain whose stationary distribution is a target density, their convergence conditions, main algorithms and diagnostics, and their contrast with variational inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - sampling
  - bayesian_inference
  - monte_carlo
semantic_relations:
  - type: relates
    links:
      - "[[metropolis_hastings]]"
      - "[[gibbs_sampling]]"
      - "[[hamiltonian_monte_carlo]]"
      - "[[detailed_balance]]"
      - "[[variational_inference]]"
      - "[[neural_sampling]]"
---

# Markov Chain Monte Carlo

## Overview
Markov chain Monte Carlo (MCMC) approximates expectations under a distribution $\pi$ that can be evaluated only up to a normalising constant, as is typical of Bayesian posteriors. A Markov chain $x_1,x_2,\dots$ is constructed so that $\pi$ is its stationary distribution, and averages along the chain approximate expectations:
$$\mathbb{E}_\pi[g]\approx\frac{1}{N}\sum_{t=1}^{N}g(x_t).$$

## Convergence
A chain with an invariant distribution $\pi$ that is irreducible and aperiodic converges to $\pi$ from any starting point, and ergodic averages converge almost surely. A sufficient way to guarantee invariance is detailed balance, $\pi(x)P(x\to x')=\pi(x')P(x'\to x)$. Successive samples are correlated, so the Monte Carlo error scales as $1/\sqrt{N_{\mathrm{eff}}}$, where the effective sample size $N_{\mathrm{eff}}$ is smaller than $N$ in proportion to the chain's autocorrelation. Early samples before the chain reaches its stationary regime are normally discarded as burn-in.

## Algorithms
- **Metropolis-Hastings** proposes a move and accepts it with a probability that preserves detailed balance.
- **Gibbs sampling** updates each variable from its conditional distribution given the others.
- **Hamiltonian Monte Carlo** uses gradients of the log density to make long, informed proposals.
- **Tempering and annealing** variants help with multimodal targets.

## Diagnostics
Convergence cannot be proved from output alone. Common checks include trace plots, autocorrelation, comparison of several chains started from dispersed points through the potential scale reduction statistic, and effective sample size estimates.

## Relation to Active Inference
MCMC is the main sampling-based alternative to variational inference. Active Inference as usually formulated uses variational methods, replacing the posterior by a tractable family and minimising free energy, which trades exactness for speed and gives a biologically interpretable message-passing scheme. MCMC appears in two other places in the vault's themes. It is a standard way to fit generative models to behavioural or neural data, and it is the basis of proposals that neural variability reflects sampling from a posterior. Whether cortical computation is better described by variational or sampling schemes remains debated, and hybrid proposals exist.

## Related concepts
- [[metropolis_hastings]] — the foundational MCMC algorithm
- [[gibbs_sampling]] — coordinate-wise sampling from conditionals
- [[hamiltonian_monte_carlo]] — gradient-based proposals
- [[importance_sampling]] — a non-Markov Monte Carlo alternative
- [[variational_inference]] — the optimisation-based counterpart
- [[neural_sampling]] — the hypothesis that neural activity implements sampling

## References
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of state calculations by fast computing machines. Journal of Chemical Physics, 21, 1087-1092.
- Robert, C. P., & Casella, G. (2004). Monte Carlo Statistical Methods (2nd ed.). Springer.
- Brooks, S., Gelman, A., Jones, G., & Meng, X.-L. (Eds.). (2011). Handbook of Markov Chain Monte Carlo. Chapman & Hall/CRC.
