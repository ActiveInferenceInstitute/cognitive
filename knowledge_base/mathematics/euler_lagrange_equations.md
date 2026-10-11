---
title: Euler-Lagrange Equations
description: The differential equations satisfied by functions that make an action functional stationary, with their derivation from the calculus of variations, standard examples, and role in path-integral formulations of free energy.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - calculus_of_variations
  - classical_mechanics
  - action_principle
semantic_relations:
  - type: relates
    links:
      - "[[variational_calculus]]"
      - "[[lagrangian_mechanics]]"
      - "[[hamiltonian_mechanics]]"
      - "[[least_action_principle]]"
      - "[[path_integral_free_energy]]"
---

# Euler-Lagrange Equations

## Overview
The Euler-Lagrange equations give necessary conditions for a function to be a stationary point of a functional. For a functional
$$
J[q]=\int_{t_0}^{t_1}L\big(t,q(t),\dot q(t)\big)\,dt
$$
with fixed endpoint values, a smooth path $q$ is stationary only if it satisfies, for each coordinate $q_i$,
$$
\frac{d}{dt}\frac{\partial L}{\partial \dot q_i}-\frac{\partial L}{\partial q_i}=0.
$$
They are derived by perturbing the path, $q\to q+\varepsilon\eta$ with $\eta(t_0)=\eta(t_1)=0$, requiring $\frac{dJ}{d\varepsilon}=0$ at $\varepsilon=0$, integrating by parts and applying the fundamental lemma of the calculus of variations. Stationary does not imply minimal; the equations identify candidates, and further conditions decide their nature.

## Examples and extensions
- **Classical mechanics**: with $L=T-V$ (kinetic minus potential energy), the equations reproduce Newton's second law, and they are the starting point of Lagrangian mechanics (Goldstein et al., 2002). Generalised coordinates make them convenient for constrained systems.
- **Geodesics**: taking $L$ as the length element in a Riemannian metric gives the geodesic equation.
- **Legendre transform**: defining momenta $p_i=\partial L/\partial\dot q_i$ yields the equivalent first-order Hamiltonian form.
- **Higher derivatives and fields**: for $L$ depending on $\ddot q$, or on fields of space and time, analogous equations hold with additional terms (Gelfand & Fomin, 1963).

## Relation to Active Inference
In continuous-time formulations the quantity minimised is the time integral of free energy, called free action, and its stationary paths are found with exactly this machinery. Path-integral treatments of the free energy principle write the probability of a trajectory in terms of an action functional, and the most probable path satisfies Euler-Lagrange equations, which have the form of equations of motion for beliefs or states. This is a mathematical equivalence once the action functional has been defined. What the action physically represents for a biological agent is a modelling interpretation, and the standard discrete-time Active Inference models make no use of these equations.

## Related concepts
- [[variational_calculus]] — the theory from which the equations derive
- [[lagrangian_mechanics]] — their original application
- [[hamiltonian_mechanics]] — the dual formulation through the Legendre transform
- [[least_action_principle]] — the physical principle they express
- [[path_integral_free_energy]] — where the equations enter free energy formulations
- [[geodesic_geometry]] — paths of extremal length

## References
- Goldstein, H., Poole, C., & Safko, J. (2002). Classical Mechanics (3rd ed.). Addison-Wesley.
- Gelfand, I. M., & Fomin, S. V. (1963). Calculus of Variations. Prentice-Hall.
- Landau, L. D., & Lifshitz, E. M. (1976). Mechanics (3rd ed.). Pergamon Press.
