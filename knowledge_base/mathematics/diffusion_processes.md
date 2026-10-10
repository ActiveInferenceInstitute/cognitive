---
title: Diffusion Processes
description: Continuous-time Markov processes with continuous paths, defined by drift and diffusion terms, and their link to the Fokker-Planck equation, Brownian motion, and stochastic dynamics in free-energy models.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - stochastic_processes
  - sde
  - fokker_planck
semantic_relations:
  - type: relates
    links:
      - "[[stochastic_processes]]"
      - "[[stochastic_differential_equations]]"
      - "[[fokker_planck]]"
      - "[[langevin_dynamics]]"
---

# Diffusion Processes

## Overview
A diffusion process is a continuous-time Markov process with almost surely continuous sample paths. It is characterised locally by two functions: a drift $b(x)$, the mean velocity, and a diffusion matrix $D(x)=\tfrac12\sigma\sigma^{\top}$, the rate at which variance accumulates. The prototype is Brownian motion, and general diffusions arise as solutions of Ito stochastic differential equations.

## Formalism
An Ito diffusion solves
$$dX_t=b(X_t)\,dt+\sigma(X_t)\,dW_t,$$
where $W_t$ is a Wiener process with independent Gaussian increments of variance $dt$. Its transition density $p(x,t)$ satisfies the Fokker-Planck (forward Kolmogorov) equation
$$\partial_t p=-\nabla\cdot(b\,p)+\nabla\cdot\!\big(\nabla\cdot(D\,p)\big),$$
which for constant isotropic $D$ reduces to $\partial_t p=-\nabla\cdot(bp)+D\nabla^2p$. The infinitesimal generator $\mathcal{L}=b\cdot\nabla+D:\nabla\nabla$ drives the backward equation for expectations.

- **Brownian motion:** $b=0$, constant $D$. In one dimension $\langle x^2\rangle=2Dt$; a simulation of 20,000 paths gives mean-square displacement within about 1% of this.
- **Ornstein-Uhlenbeck process:** $dX=-\theta X\,dt+\sigma dW$, with stationary variance $\sigma^2/(2\theta)$. It models mean-reverting dynamics and linear Gaussian state-space models.
- **Stationary density.** For gradient drift $b=-D\nabla V/k$ and constant $D$, the stationary law is Boltzmann-like, $p^*\propto e^{-V/k}$. With a non-gradient (solenoidal) component in $b$, the stationary density persists but the current is non-zero, which is the non-equilibrium steady state case.
- **Score-based generative models** run a diffusion forward to noise and learn its time reversal, an application of the same machinery.

## Role in Active Inference
The free-energy principle models the world and agent as a random dynamical system, usually a Langevin equation $\dot x=f(x)+\omega$ with Gaussian fluctuations $\omega$, which is a diffusion. The steady-state density $p^*$ then obeys Fokker-Planck, and the decomposition of $f$ into dissipative and solenoidal parts is read off from the stationary condition ([[non_equilibrium_steady_state]]). In path-integral treatments the diffusion tensor $D$ sets the weight of path fluctuations in the action ([[path_integral_free_energy]]). In cognitive modelling, drift-diffusion models of evidence accumulation use a one-dimensional diffusion with absorbing boundaries to describe choice and reaction time ([[evidence_accumulation]]).

## Related concepts
- [[stochastic_differential_equations]] — the equations whose solutions are diffusions
- [[fokker_planck]] — evolution of the density
- [[langevin_dynamics]] — physical diffusion with friction and noise
- [[fluctuation_dissipation]] — links noise strength to dissipation
- [[evidence_accumulation]] — drift-diffusion models of decisions
- [[boundary_conditions]] — absorbing and reflecting edges for diffusing densities
