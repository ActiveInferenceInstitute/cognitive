---
title: Stochastic Calculus
description: The calculus of integrals and differential equations driven by Brownian motion, including the Ito integral, Ito's lemma, stochastic differential equations and the Fokker-Planck equation, with uses in continuous-time Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - stochastic_processes
  - differential_equations
  - continuous_time
semantic_relations:
  - type: relates
    links:
      - "[[ito_calculus]]"
      - "[[wiener_process]]"
      - "[[stochastic_differential_equations]]"
      - "[[fokker_planck]]"
      - "[[langevin_dynamics]]"
      - "[[martingale_theory]]"
---

# Stochastic Calculus

## Overview
Stochastic calculus extends integration and differentiation to processes with random, nowhere-differentiable paths. Its central object is the Wiener process $W_t$, or Brownian motion, whose increments over disjoint intervals are independent and Gaussian with variance equal to the interval length. Because its paths have unbounded variation, ordinary Riemann-Stieltjes integration fails, and a new definition of the integral against $W$ is required.

## Ito integral and lemma
The Ito integral $\int_0^t H_s\,dW_s$ is defined for processes $H$ that do not anticipate the future, as the limit of sums that evaluate the integrand at the left end of each subinterval. The result is a martingale. A distinctive feature is the quadratic variation of Brownian motion, which is expressed informally as $(dW_t)^2=dt$. Consequently the chain rule acquires a second-order term. For the stochastic differential equation $dX_t=a(X_t)\,dt+b(X_t)\,dW_t$ and a smooth function $f$, Ito's lemma gives
$$df(X_t)=\Big(a f'+\tfrac12 b^2 f''\Big)dt+b f'\,dW_t.$$
Evaluating the integrand at the midpoint instead gives the Stratonovich integral, which obeys the ordinary chain rule. The two conventions differ by a drift correction when $b$ depends on $X$, and the choice must follow from the modelling context.

## Density evolution and simulation
The probability density $p(x,t)$ of the solution obeys the Fokker-Planck equation,
$$\partial_t p=-\partial_x(a\,p)+\tfrac12\,\partial_x^2(b^2p).$$
Sample paths are simulated by the Euler-Maruyama scheme $X_{n+1}=X_n+a\,\Delta t+b\sqrt{\Delta t}\,\xi_n$ with standard normal $\xi_n$, which converges with strong order one half.

## Relation to Active Inference
Continuous-time formulations of the free energy principle and Active Inference describe the dynamics of states with Langevin equations, a stochastic differential equation with a deterministic flow plus random fluctuations, and describe the associated densities with the Fokker-Planck equation. Several results about steady-state densities, flows with solenoidal and dissipative components, and Markov blankets are derived with these tools. Predictive coding schemes in generalised coordinates typically assume smooth fluctuations and are interpreted in the Stratonovich sense. The calculus is therefore a direct mathematical foundation for continuous-time versions of the theory.

## Related concepts
- [[wiener_process]] — the driving noise of stochastic integrals
- [[ito_calculus]] — the specific calculus with the second-order correction
- [[stochastic_differential_equations]] — equations defined through these integrals
- [[fokker_planck]] — evolution equation for the density of the solution
- [[langevin_dynamics]] — noisy gradient dynamics widely used in physics and sampling

## References
- Øksendal, B. (2003). Stochastic Differential Equations: An Introduction with Applications (6th ed.). Springer.
- Gardiner, C. W. (2009). Stochastic Methods: A Handbook for the Natural and Social Sciences (4th ed.). Springer.
- Karatzas, I., & Shreve, S. E. (1991). Brownian Motion and Stochastic Calculus (2nd ed.). Springer.
