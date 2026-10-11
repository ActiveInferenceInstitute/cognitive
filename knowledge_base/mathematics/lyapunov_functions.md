---
title: Lyapunov Functions
description: Scalar functions that decrease along trajectories and certify stability of an equilibrium without solving the dynamics, with the standard conditions, the quadratic case, and the link to free energy.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - stability
  - dynamical_systems
  - control_theory
semantic_relations:
  - type: relates
    links:
      - "[[lyapunov_stability]]"
      - "[[stability_analysis]]"
      - "[[dynamical_systems]]"
      - "[[gradient_flows]]"
      - "[[free_energy_principle]]"
---

# Lyapunov Functions

## Overview
A Lyapunov function is a scalar function of the state that acts as a generalised energy: it is bounded below and does not increase along trajectories. Finding one proves that an equilibrium is stable without integrating the equations of motion. This is Lyapunov's direct method.

## Conditions
For $\dot x=f(x)$ with an equilibrium at $x^*$, suppose $V$ is continuously differentiable, $V(x^*)=0$ and $V(x)>0$ for $x\neq x^*$ in a neighbourhood. Its derivative along trajectories is $\dot V=\nabla V\cdot f$.
- If $\dot V\le 0$, the equilibrium is Lyapunov stable.
- If $\dot V<0$ for $x\neq x^*$, it is asymptotically stable.
- If $V\to\infty$ as $\|x\|\to\infty$ (radial unboundedness) and $\dot V<0$ everywhere else, the equilibrium is globally asymptotically stable.

LaSalle's invariance principle relaxes strict negativity: trajectories converge to the largest invariant set contained in $\{\dot V=0\}$. For a linear system $\dot x=Ax$, a quadratic function $V=x^\top Px$ works exactly when $A^\top P+PA=-Q$ has a positive definite solution $P$ for some positive definite $Q$, which holds if and only if all eigenvalues of $A$ have negative real part. Converse theorems show that Lyapunov functions exist for asymptotically stable systems, but they give no recipe for constructing them. A gradient system $\dot x=-\nabla V(x)$ has $V$ as a Lyapunov function, since $\dot V=-\|\nabla V\|^2\le 0$.

## Uses
Lyapunov functions are used to design stabilising controllers, to estimate regions of attraction (see [[basin_of_attraction]]), and to prove convergence of optimisation and learning dynamics.

## Relation to Active Inference
In the free energy principle, a system with a non-equilibrium steady-state density $p(x)$ has a flow that can be split into a dissipative part, which descends the surprisal $-\ln p(x)$, and a solenoidal part, which circulates on its contours. For the deterministic component of this flow, surprisal is non-increasing, so it plays the role of a Lyapunov function with the steady state at its minimum. Variational free energy is then a bound on surprisal that agents can evaluate. This follows from the stated decomposition, but it applies to the mean flow and depends on the steady-state and Markov-blanket assumptions of the formalism, which are debated for real biological systems.

## Related concepts
- [[lyapunov_stability]] — the notion of stability the functions certify
- [[lyapunov_exponents]] — a different, trajectory-based measure of divergence
- [[gradient_flows]] — flows with a built-in Lyapunov function
- [[non_equilibrium_steady_state]] — the steady-state density underlying the decomposition
- [[control_theory]] — where Lyapunov design is standard practice
- [[free_energy_principle]] — surprisal and free energy as Lyapunov-like quantities

## References
- Khalil, H. K. (2002). Nonlinear Systems (3rd ed.). Prentice Hall.
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
- Friston, K. (2013). Life as we know it. Journal of the Royal Society Interface, 10, 20130475.
