---
title: Lyapunov Stability
description: The standard definitions of stable and asymptotically stable equilibria, the direct method based on Lyapunov functions, and the use of free energy as a Lyapunov function in Active Inference dynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - stability
  - control_theory
semantic_relations:
  - type: relates
    links:
      - "[[lyapunov_functions]]"
      - "[[lyapunov_exponents]]"
      - "[[dynamical_systems]]"
      - "[[structural_stability]]"
      - "[[gradient_flows]]"
      - "[[free_energy_principle]]"
wikidata: Q1341651
---

# Lyapunov Stability

## Overview
Lyapunov stability describes how solutions of a dynamical system behave when started near an equilibrium. For $\dot{x}=f(x)$ with equilibrium $x^*$ satisfying $f(x^*)=0$, the equilibrium is stable if for every $\varepsilon>0$ there is a $\delta>0$ such that $\lVert x(0)-x^*\rVert<\delta$ implies $\lVert x(t)-x^*\rVert<\varepsilon$ for all $t\ge0$. It is asymptotically stable if, in addition, solutions starting close enough converge to $x^*$. It is unstable if it is not stable.

## The direct method
The direct method avoids solving the equations. A function $V$ that is continuously differentiable, positive definite around $x^*$ (that is, $V(x^*)=0$ and $V(x)>0$ elsewhere nearby) and has non-positive derivative along trajectories,
$$\dot{V}(x)=\nabla V(x)\cdot f(x)\le 0,$$
proves stability. If $\dot V<0$ away from $x^*$, the equilibrium is asymptotically stable. LaSalle's invariance principle extends this: if $\dot V\le0$ and the only solution that stays in the set where $\dot V=0$ is the equilibrium itself, convergence still follows. If $V$ is also radially unbounded, the conclusion holds globally. The indirect method linearises the system: the equilibrium is asymptotically stable if every eigenvalue of the Jacobian $Df(x^*)$ has negative real part and unstable if any has positive real part. Eigenvalues on the imaginary axis leave the question undecided.

## Constructing functions
No general recipe exists for finding $V$. Energy is the natural candidate in mechanical systems. In gradient systems $\dot{x}=-\nabla U(x)$, the potential $U$ itself works, since $\dot U=-\lVert\nabla U\rVert^2\le0$. For linear systems $\dot{x}=Ax$, a quadratic $V=x^\top Px$ is obtained by solving the Lyapunov equation $A^\top P+PA=-Q$.

## Relation to Active Inference
The link is explicit in the dynamics of belief updating. When internal states perform a gradient descent on variational free energy $F$, $\dot\mu=-\nabla_\mu F$, free energy is a Lyapunov function of that flow and $\dot F=-\lVert\nabla_\mu F\rVert^2\le0$. More generally, the free energy principle describes flows in which surprisal or free energy decreases on average, and it uses Lyapunov-type arguments to say why a system stays near its attracting set. These results hold for the stated flows. For noisy flows the statement applies to expected values, and for real systems it is a modelling assumption.

## Related concepts
- [[lyapunov_functions]] — the scalar functions that certify stability
- [[lyapunov_exponents]] — quantitative rates of convergence and divergence
- [[gradient_flows]] — flows for which a potential is a natural Lyapunov function
- [[attractor_dynamics]] — attracting sets whose stability is analysed this way
- [[structural_stability]] — robustness of dynamics to perturbations of the equations

## References
- Khalil, H. K. (2002). Nonlinear Systems (3rd ed.). Prentice Hall.
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
- Lyapunov, A. M. (1992). The General Problem of the Stability of Motion. Taylor & Francis.
