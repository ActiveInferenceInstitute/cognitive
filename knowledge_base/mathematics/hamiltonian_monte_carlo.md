---
title: Hamiltonian Monte Carlo
description: A Markov chain Monte Carlo method that uses Hamiltonian dynamics with auxiliary momentum to propose distant states, its leapfrog integrator, tuning, and the shared mechanics with physics-based formulations of inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - monte_carlo
  - sampling
  - hamiltonian_dynamics
semantic_relations:
  - type: relates
    links:
      - "[[monte_carlo_methods]]"
      - "[[hamiltonian_mechanics]]"
      - "[[langevin_dynamics]]"
      - "[[symplectic_geometry]]"
      - "[[bayesian_inference]]"
aliases:
  - hamiltonian_mc
  - hybrid_monte_carlo
wikidata: Q1639846
---

# Hamiltonian Monte Carlo

## Overview
Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo algorithm for sampling from a density $\pi(q)\propto e^{-U(q)}$ over continuous variables, such as a Bayesian posterior with $U(q)=-\ln\pi(q)$. It was introduced as hybrid Monte Carlo by Duane, Kennedy, Pendleton and Roweth (1987) for lattice field theory and later adopted widely in statistics. By following simulated physical trajectories it makes large moves that still have a high acceptance probability, which avoids the slow random-walk behaviour of simple Metropolis proposals.

## Mechanism
The position $q$ is augmented with an auxiliary momentum $p$ drawn from a Gaussian $\mathcal{N}(0,M)$. Together they define a Hamiltonian

$$H(q,p)=U(q)+\tfrac12 p^{\top}M^{-1}p,$$

and the dynamics $\dot q=\partial H/\partial p$, $\dot p=-\partial H/\partial q$ conserve $H$, preserve volume and are reversible (see [[hamiltonian_mechanics]]). One iteration has three parts:
1. Resample the momentum $p\sim\mathcal{N}(0,M)$.
2. Integrate the dynamics for $L$ steps of size $\epsilon$ using the leapfrog scheme, a symplectic integrator that keeps the energy error bounded (see [[symplectic_geometry]]).
3. Accept or reject the endpoint with probability $\min\{1,\exp(H_{\text{start}}-H_{\text{end}})\}$, which corrects for integration error and keeps the target distribution exactly invariant.

The gradient of $U$ is needed at each step, so the density must be differentiable. Performance depends on the step size, trajectory length and mass matrix $M$. A trajectory that is too short behaves like a random walk, whereas one that is too long can loop back. The No-U-Turn Sampler adapts the trajectory length automatically and is the default in several probabilistic programming systems. Posteriors with sharply varying curvature, such as hierarchical models with funnel geometry, remain difficult.

## Relation to Active Inference
HMC is a tool for inference, not part of the Active Inference formalism, which typically uses variational approximations in place of sampling. The connection is mathematical. Both sampling and the continuous-time formulations of the free energy principle draw on [[langevin_dynamics]] and Hamiltonian structure, and the vault's path integral material discusses these mechanics. HMC can serve as a reference method for checking variational posteriors in models where exact answers are not available. No claim follows that neural systems implement Hamiltonian sampling.

## Related concepts
- [[monte_carlo_methods]] — the family HMC belongs to
- [[hamiltonian_mechanics]] — the dynamics HMC borrows
- [[langevin_dynamics]] — a related gradient-based sampler with noise
- [[symplectic_geometry]] — the geometry behind the leapfrog integrator
- [[bayesian_inference]] — the typical application, sampling posteriors

## References
- Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D. (1987). Hybrid Monte Carlo. Physics Letters B, 195, 216-222.
- Neal, R. M. (2011). MCMC using Hamiltonian dynamics. In S. Brooks, A. Gelman, G. Jones, & X.-L. Meng (Eds.), Handbook of Markov Chain Monte Carlo (pp. 113-162). Chapman and Hall/CRC.
