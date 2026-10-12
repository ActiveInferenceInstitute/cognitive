---
title: Brownian Motion
description: The random motion of suspended particles and its mathematical model, the Wiener process, covering defining properties, diffusion scaling, Langevin dynamics, and its use as noise in the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - stochastic_processes
  - diffusion
  - physics
semantic_relations:
  - type: relates
    links:
      - "[[wiener_process]]"
      - "[[stochastic_differential_equations]]"
      - "[[langevin_dynamics]]"
      - "[[diffusion_processes]]"
      - "[[random_walk]]"
      - "[[stochastic_calculus]]"
---

# Brownian Motion

## Overview
Brownian motion is the irregular movement of small particles suspended in a fluid, observed by Robert Brown in 1827 and explained by Einstein (1905) as the result of collisions with the molecules of the fluid. The name is also used for its idealised mathematical model, the standard Brownian motion or Wiener process. The mathematical object is treated in more detail on the [[wiener_process]] page.

## Mathematical definition
A standard one-dimensional Brownian motion $(W_t)_{t \geq 0}$ is a stochastic process with these properties:
- $W_0 = 0$.
- Increments are independent: for $0 \leq s < t$, $W_t - W_s$ is independent of the history up to time $s$.
- Increments are Gaussian: $W_t - W_s \sim \mathcal{N}(0, t - s)$.
- Sample paths are continuous.

Consequently $\mathrm{Var}(W_t) = t$ and $\mathrm{Cov}(W_s, W_t) = \min(s, t)$. The process has the scaling property that $W_{ct}$ has the same distribution as $\sqrt{c}\,W_t$, so displacement grows like $\sqrt{t}$. Almost surely, sample paths are nowhere differentiable and have unbounded variation, but their quadratic variation over $[0, t]$ equals $t$, which is the basis of the Itô calculus (see [[stochastic_calculus]]). It is the continuous-time limit of a scaled simple [[random_walk]].

## Physical and dynamical use
For a particle in one dimension, the mean squared displacement is $\langle x^2 \rangle = 2Dt$, where $D$ is the diffusion coefficient. Einstein related $D$ to temperature and friction, and for a sphere of radius $a$ in a fluid of viscosity $\eta$ the Stokes-Einstein relation gives $D = k_B T / (6\pi\eta a)$. Adding deterministic drift gives the overdamped Langevin equation $dx = -\nabla U(x)\,dt + \sqrt{2D}\,dW_t$ (in units where the friction coefficient is one), and the probability density of $x$ then satisfies a Fokker-Planck equation (see [[langevin_dynamics]]).

## Relation to Active Inference
The free energy principle is stated for systems described by Langevin or stochastic differential equations, $\dot{x} = f(x) + \omega$, in which the random fluctuations $\omega$ are the formal derivative of Wiener processes (see [[stochastic_differential_equations]]). The noise is taken to be Gaussian and white, and the derivation then assumes a nonequilibrium steady-state density, from which a Markov blanket partition and an interpretation of the dynamics as a flow on surprisal are obtained. Continuous-time generative models in Active Inference also commonly assume smooth random fluctuations, which are analytic and are not Brownian, because the generalised coordinates of motion need differentiable noise. The role of Brownian motion is thus foundational for the physics but not for every model.

## Related concepts
- [[wiener_process]] — the mathematical model in detail
- [[stochastic_differential_equations]] — equations driven by Brownian noise
- [[langevin_dynamics]] — dynamics with friction and thermal noise
- [[diffusion_processes]] — the broader family of processes
- [[random_walk]] — the discrete-time precursor
- [[fokker_planck]] — the equation for the evolution of the density

## References
- Einstein, A. (1905). Über die von der molekularkinetischen Theorie der Wärme geforderte Bewegung von in ruhenden Flüssigkeiten suspendierten Teilchen. Annalen der Physik, 17, 549-560.
- Karatzas, I., & Shreve, S. E. (1991). Brownian Motion and Stochastic Calculus (2nd ed.). Springer.
- Øksendal, B. (2003). Stochastic Differential Equations: An Introduction with Applications (6th ed.). Springer.
