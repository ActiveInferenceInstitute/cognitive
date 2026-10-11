---
title: Metropolis-Hastings Algorithm
description: The accept-reject Markov chain Monte Carlo algorithm that samples from an unnormalised target density, its acceptance rule, detailed balance proof sketch, tuning guidance and place among sampling methods.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - sampling
  - monte_carlo
  - bayesian_inference
semantic_relations:
  - type: relates
    links:
      - "[[markov_chain_monte_carlo]]"
      - "[[detailed_balance]]"
      - "[[gibbs_sampling]]"
      - "[[hamiltonian_monte_carlo]]"
      - "[[importance_sampling]]"
---

# Metropolis-Hastings Algorithm

## Overview
The Metropolis-Hastings algorithm generates a Markov chain whose stationary distribution is a target density $\pi(x)$ that is known only up to a constant, for example an unnormalised posterior. It was introduced by Metropolis and colleagues for physical systems and generalised by Hastings to asymmetric proposals.

## The algorithm
From the current state $x$, draw a candidate $x'$ from a proposal distribution $q(x'\mid x)$. Accept it with probability
$$\alpha(x\to x')=\min\left\{1,\;\frac{\pi(x')\,q(x\mid x')}{\pi(x)\,q(x'\mid x)}\right\}.$$
If the candidate is rejected, the chain stays at $x$ and the repeated value is retained in the sample. Because only the ratio $\pi(x')/\pi(x)$ appears, the normalising constant cancels. When $q$ is symmetric, $q(x'\mid x)=q(x\mid x')$, the rule reduces to the original Metropolis form $\min\{1,\pi(x')/\pi(x)\}$.

## Why it works
For $x\ne x'$ the transition density is $P(x\to x')=q(x'\mid x)\,\alpha(x\to x')$. Substituting the acceptance rule shows
$$\pi(x)\,q(x'\mid x)\,\alpha(x\to x')=\pi(x')\,q(x\mid x')\,\alpha(x'\to x),$$
which is detailed balance. Detailed balance implies that $\pi$ is invariant, and under mild irreducibility and aperiodicity conditions the chain converges to $\pi$.

## Practical considerations
Performance depends on the proposal. A random-walk proposal with step size that is too small accepts often but explores slowly, and one that is too large is rejected most of the time. Under idealised assumptions in high dimensions, an acceptance rate of about 0.234 is asymptotically optimal for random-walk Metropolis, and tuning toward that range is common. Multimodal targets can trap the chain, which motivates tempering. Gradient-based proposals as used in Hamiltonian Monte Carlo scale better with dimension. Convergence has to be assessed with diagnostics, because acceptance rates alone do not show it.

## Relation to Active Inference
Metropolis-Hastings is not part of the Active Inference formalism, which relies on variational optimisation. It appears in the surrounding methodology, for example when the parameters of a generative model are fitted to behavioural data by sampling, and in discussions of whether neural dynamics implement sampling from beliefs. The link is therefore methodological and not a derivation.

## Related concepts
- [[markov_chain_monte_carlo]] — the family of methods to which the algorithm belongs
- [[detailed_balance]] — the condition that guarantees the correct stationary distribution
- [[gibbs_sampling]] — a special case with acceptance probability one
- [[hamiltonian_monte_carlo]] — proposals built from gradient information
- [[annealing_schedules]] — temperature schedules used for multimodal targets

## References
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of state calculations by fast computing machines. Journal of Chemical Physics, 21, 1087-1092.
- Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. Biometrika, 57, 97-109.
- Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. Annals of Applied Probability, 7, 110-120.
