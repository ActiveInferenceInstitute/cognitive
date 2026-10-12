---
title: Symplectic Integrators
description: Numerical schemes for Hamiltonian systems that preserve the symplectic structure of phase space, giving long-term stability, with the leapfrog method as the principal example and its use in Hamiltonian Monte Carlo.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - numerical_methods
  - hamiltonian_mechanics
  - geometric_integration
semantic_relations:
  - type: relates
    links:
      - "[[hamiltonian_mechanics]]"
      - "[[symplectic_geometry]]"
      - "[[hamiltonian_monte_carlo]]"
      - "[[variational_integrators]]"
      - "[[numerical_methods]]"
---

# Symplectic Integrators

## Overview
A symplectic integrator is a numerical method for Hamiltonian equations of motion whose one-step map is a symplectic transformation. The exact flow of a Hamiltonian system has this property: its Jacobian $\Phi'$ satisfies $\Phi'^\top J\,\Phi'=J$ with $J$ the standard symplectic matrix, which implies conservation of phase-space volume and of the Poincare invariants. General-purpose methods such as explicit Runge-Kutta schemes break this structure and produce energy drift over long runs, while symplectic methods do not.

## The leapfrog method
For a separable Hamiltonian $H(q,p)=\tfrac12p^\top M^{-1}p+V(q)$ with step size $h$, the Stormer-Verlet (leapfrog) scheme is
$$p_{n+1/2}=p_n-\tfrac h2\nabla V(q_n),\quad q_{n+1}=q_n+h\,M^{-1}p_{n+1/2},\quad p_{n+1}=p_{n+1/2}-\tfrac h2\nabla V(q_{n+1}).$$
Each substep is a shear and therefore symplectic, so the composition is symplectic. The scheme is second-order accurate, time-reversible and requires one new gradient evaluation per step. The first-order symplectic Euler method, which updates momentum and then position using the new momentum, is a simpler member of the family. Higher-order symplectic schemes are built by composition of lower-order ones.

## Why structure preservation helps
Symplectic integrators do not conserve the energy $H$ exactly. Backward error analysis shows that, for small enough $h$, the numerical solution is very close to the exact flow of a modified Hamiltonian $\tilde H=H+h^2H_2+\cdots$, with the difference exponentially small in $1/h$ over exponentially long times for analytic systems (Hairer, Lubich & Wanner, 2006). The consequence is that the energy error stays bounded and oscillatory without systematic drift, whereas a non-symplectic method of the same order typically drifts. This holds for fixed step size. Variable step sizes can destroy the property unless handled with care.

## Use in sampling
Hamiltonian Monte Carlo proposes moves by integrating Hamiltonian dynamics with potential $V=-\ln p(q)$ using leapfrog steps. Volume preservation and reversibility ensure that the Metropolis acceptance probability depends only on the energy error, so acceptance remains high even for long trajectories (Neal, 2011). See [[hamiltonian_monte_carlo]] and [[markov_chain_monte_carlo]].

## Relation to Active Inference
The relation is through tools rather than theory. Sampling-based approximations to posterior beliefs, and simulations of generative processes with conservative Hamiltonian structure, use symplectic or HMC methods. Treatments of the [[free_energy_principle]] in generalised coordinates describe flows with solenoidal, energy-conserving components, for which structure-preserving discretisation is the natural choice, but Active Inference software does not usually require it.

## Related concepts
- [[hamiltonian_mechanics]] — the dynamics being integrated
- [[symplectic_geometry]] — the geometry preserved
- [[hamiltonian_monte_carlo]] — the main sampling application
- [[variational_integrators]] — integrators derived from discretised action principles
- [[hamiltonian_flows]] — the continuous flows approximated
- [[numerical_methods]] — the wider field

## References
- Hairer, E., Lubich, C., & Wanner, G. (2006). Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations (2nd ed.). Springer.
- Leimkuhler, B., & Reich, S. (2004). Simulating Hamiltonian Dynamics. Cambridge University Press.
- Neal, R. M. (2011). MCMC using Hamiltonian dynamics. In S. Brooks, A. Gelman, G. Jones, & X.-L. Meng (Eds.), Handbook of Markov Chain Monte Carlo (pp. 113-162). Chapman & Hall/CRC.
