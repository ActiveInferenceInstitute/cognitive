---
title: Rate Equations
description: Differential equations describing how quantities such as concentrations or populations change with time, including first-order decay, mass-action kinetics, Michaelis-Menten kinetics, logistic growth and their steady states.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - kinetics
  - differential_equations
  - mathematical_biology
semantic_relations:
  - type: relates
    links:
      - "[[differential_equations]]"
      - "[[dynamical_systems]]"
      - "[[linear_stability]]"
      - "[[lotka_volterra]]"
      - "[[kinetics]]"
---

# Rate Equations

## Overview
A rate equation expresses the time derivative of a quantity, such as the concentration of a chemical species or the size of a population, as a function of the current state. In chemistry and biology they describe the average behaviour of many individual events in the large-number limit, so that stochastic reactions become deterministic ordinary differential equations. Their structure is that of an autonomous dynamical system $\dot x=g(x)$, studied through fixed points, stability and long-time behaviour.

## Standard forms
- **First-order decay.** $\dot N=-kN$ has solution $N(t)=N_0e^{-kt}$ and a half-life $\ln 2/k$. Radioactive decay and many first-order reactions follow it.
- **Mass-action kinetics.** For an elementary reaction $A+B\to C$ with rate constant $k$, the rate is $\tfrac{d[C]}{dt}=k[A][B]$, proportional to the product of the reactant concentrations. A reaction network gives a system of coupled equations, $\dot x=S\,v(x)$, where $S$ is the stoichiometric matrix and $v$ the vector of reaction rates.
- **Michaelis-Menten kinetics.** For an enzyme converting substrate $S$, under a quasi-steady-state approximation for the enzyme-substrate complex, the reaction velocity is $v=V_{\max}[S]/(K_M+[S])$. At low substrate it is linear in $[S]$, and at high substrate it saturates at $V_{\max}$.
- **Logistic growth.** $\dot N=rN(1-N/K)$ has a growth rate $r$ and carrying capacity $K$, with an unstable fixed point at $0$ and a stable one at $K$. Interacting populations, as in the Lotka-Volterra equations, extend this to coupled systems.

A fixed point $x^*$ satisfies $g(x^*)=0$ and is locally stable if the Jacobian at $x^*$ has eigenvalues with negative real part. Widely separated rates produce stiff systems that need implicit solvers. Rate equations are valid when fluctuations are small. For small molecule counts, the chemical master equation is the appropriate description and the rate equation is recovered for the mean in the large-volume limit under suitable conditions.

## Relation to Active Inference
Rate equations are standard tools in systems biology, but they are not part of Active Inference as such. The link is formal: neural-mass and gradient-flow models used with the free energy principle are also systems of first-order differential equations, so the same tools of fixed-point and stability analysis apply. Active Inference adds a specific form for the flow, descent on a free energy with additional solenoidal components.

## Related concepts
- [[differential_equations]] — the mathematical class of rate equations
- [[dynamical_systems]] — fixed points, stability and long-time behaviour
- [[linear_stability]] — Jacobian-based analysis near a fixed point
- [[lotka_volterra]] — coupled population rate equations
- [[kinetics]] — rates of change in physical and chemical systems
- [[complex_systems_biology]] — biological networks modelled with rate equations

## References
- Murray, J. D. (2002). Mathematical Biology I: An Introduction (3rd ed.). Springer.
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
- Michaelis, L., & Menten, M. L. (1913). Die Kinetik der Invertinwirkung. Biochemische Zeitschrift, 49, 333-369.
