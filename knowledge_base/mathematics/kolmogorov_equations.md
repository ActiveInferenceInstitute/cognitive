---
title: Kolmogorov Equations
description: The forward and backward equations describing how transition probabilities of a Markov process evolve, in discrete-state and diffusion forms, including the Fokker-Planck equation and its role in the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - stochastic_processes
  - master_equation
  - fokker_planck
semantic_relations:
  - type: relates
    links:
      - "[[fokker_planck]]"
      - "[[stochastic_differential_equations]]"
      - "[[stochastic_processes]]"
      - "[[markov_properties]]"
      - "[[non_equilibrium_steady_state]]"
---

# Kolmogorov Equations

## Overview
The Kolmogorov equations govern the time evolution of transition probabilities of a continuous-time Markov process. They were introduced by Kolmogorov (1931) and come in two forms. The forward equation describes how the distribution of the current state changes, with the initial state fixed. The backward equation describes how the probability of a future event depends on the initial state, and acts on functions of the starting point.

## Discrete and continuous forms
For a continuous-time Markov chain on a finite or countable state space with generator (rate) matrix $Q$, whose off-diagonal entries are transition rates and whose rows sum to zero, the transition matrix $P(t)$ satisfies
$$
\frac{d}{dt}P(t)=P(t)\,Q\ \ (\text{forward}),\qquad \frac{d}{dt}P(t)=Q\,P(t)\ \ (\text{backward}).
$$
The forward equation applied to a probability vector is the master equation, $\dot p_i=\sum_{j}(p_jQ_{ji})$.

For a diffusion $dX=f(X)\,dt+\sigma(X)\,dW$ with diffusion tensor $D=\tfrac12\sigma\sigma^{\top}$, the density $p(x,t)$ obeys the forward equation, known as the Fokker-Planck equation,
$$
\partial_t p=-\nabla\!\cdot\!(f\,p)+\nabla\!\cdot\!\nabla\!\cdot\!(D\,p),
$$
where the second term is the double divergence $\sum_{ij}\partial_i\partial_j(D_{ij}p)$. The backward equation, for $u(x,t)=\mathbb{E}[\phi(X_T)\mid X_t=x]$, is $-\partial_t u=f\cdot\nabla u+\sum_{ij}D_{ij}\partial_i\partial_j u$. Both are linear equations in their unknowns, and stationary solutions of the forward equation are the invariant densities (Gardiner, 2009).

## Relation to Active Inference
The link is a core mathematical tool of the free energy principle. The dynamics of a system's states are modelled by a Langevin equation, and the Fokker-Planck equation determines how the density over states evolves. Assuming a non-equilibrium steady-state density, the flow can be decomposed into a gradient part that descends the surprise $-\ln p$ and a solenoidal part that circulates along its level sets. Showing that internal states appear to infer external states in this setting relies on this stationary solution. The equations themselves are standard stochastic analysis, whereas interpreting the stationary density as a generative model is part of the framework's contested modelling claims.

## Related concepts
- [[fokker_planck]] — the diffusion form of the forward equation
- [[stochastic_differential_equations]] — the sample-path description of the same processes
- [[stochastic_processes]] — the general class of random dynamics
- [[markov_properties]] — the property that makes the equations closed
- [[non_equilibrium_steady_state]] — stationary solutions of the forward equation
- [[langevin_dynamics]] — the noisy dynamics whose density is evolved

## References
- Kolmogorov, A. N. (1931). Über die analytischen Methoden in der Wahrscheinlichkeitsrechnung. Mathematische Annalen, 104, 415-458.
- Gardiner, C. W. (2009). Handbook of Stochastic Methods (4th ed.). Springer.
- Øksendal, B. (2003). Stochastic Differential Equations: An Introduction with Applications (6th ed.). Springer.
