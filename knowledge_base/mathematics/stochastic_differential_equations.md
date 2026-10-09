---
title: Stochastic Differential Equations
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - sde
  - ito_calculus
  - langevin
  - fokker_planck
semantic_relations:
  - type: foundation
    links:
      - "[[stochastic_processes]]"
      - "[[differential_equations]]"
  - type: relates
    links:
      - "[[langevin_dynamics]]"
      - "[[fokker_planck]]"
      - "[[non_equilibrium_steady_state]]"
---

# Stochastic Differential Equations

## Overview
A stochastic differential equation (SDE) augments an ordinary differential equation with a random forcing term, modelling systems driven by noise. The forcing is typically white noise, made rigorous as increments of a Wiener process $W_t$.

## Definition / Formalism
In Ito form,

$$dX_t=f(X_t,t)\,dt+g(X_t,t)\,dW_t,$$

where $f$ is the drift, $g$ the diffusion coefficient and $dW_t$ has variance $dt$. Ito's lemma replaces the ordinary chain rule: for smooth $h$, $dh=\big(h_xf+\tfrac12g^2h_{xx}\big)dt+g\,h_x\,dW$. The Stratonovich interpretation obeys the ordinary chain rule but differs from Ito by a drift correction when $g$ depends on $X$. Numerically, the Euler-Maruyama scheme is $X_{n+1}=X_n+f\,\Delta t+g\,\sqrt{\Delta t}\,\xi_n$ with $\xi_n\sim\mathcal N(0,1)$; it has strong order $1/2$ and weak order $1$.

The density $p(x,t)$ evolves by the Fokker-Planck equation $\partial_tp=-\partial_x(fp)+\tfrac12\partial_x^2(g^2p)$. Example: the Ornstein-Uhlenbeck process $dX=-\theta X\,dt+\sigma\,dW$ has stationary distribution $\mathcal N\big(0,\sigma^2/(2\theta)\big)$; a simulation with $\theta=2$, $\sigma=0.8$ gives sample variance $0.165$ against the theoretical $0.16$. For multivariate gradient-type SDEs with state-independent noise covariance $2\Gamma$, a stationary density $p^*$ exists with drift $f=(Q-\Gamma)\nabla\mathcal I$, where $\mathcal I=-\ln p^*$ and $Q$ is antisymmetric: the symmetric part pulls the state toward high density and the antisymmetric part produces solenoidal flow around density contours without changing $p^*$.

## Role in Active Inference
SDEs are the dynamical language of the free energy principle. The state of a system with a Markov blanket is modelled by a Langevin equation, and the existence of a non-equilibrium steady state with density $p^*$ gives the decomposition above, from which the claim that internal states perform inference on external states is derived under additional assumptions about the blanket. In continuous-time generative models the fluctuations $w,z$ are modelled as smooth noise handled in generalized coordinates rather than as white Ito noise, which is a modelling choice for tractability. The mathematical results above are standard; the interpretation of drift as "inference" is specific to the free energy principle and remains debated.

## Related concepts
- [[stochastic_processes]] — the general theory of random processes
- [[langevin_dynamics]] — noise-driven gradient and Hamiltonian dynamics
- [[fokker_planck]] — evolution equation for the density
- [[non_equilibrium_steady_state]] — stationary density with persistent flow
- [[continuous_time_models]] — active inference models formulated with these dynamics
- [[path_integral]] — probability measures over trajectories of an SDE

## References
- Oksendal, Stochastic Differential Equations: An Introduction with Applications.
- Gardiner, Stochastic Methods: A Handbook for the Natural and Social Sciences.
- Kloeden & Platen, Numerical Solution of Stochastic Differential Equations.
