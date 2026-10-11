---
title: Field Theory
description: Classical field theory, in which dynamics are given by fields over space and time derived from an action principle, with Noether conservation laws and neural-field models, and its relation to Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - physics
  - variational_calculus
  - neural_fields
semantic_relations:
  - type: relates
    links:
      - "[[quantum_field_theory]]"
      - "[[variational_calculus]]"
      - "[[lagrangian_mechanics]]"
      - "[[path_integral]]"
aliases:
  - classical_field_theory
---

# Field Theory

## Overview
In this page field theory means classical field theory: the description of a system by fields $\phi(x,t)$, functions on space and time, with local dynamics. The quantum version, in which fields become operators or are integrated over, is covered in [[quantum_field_theory]]. The tools are the same calculus of variations used for particles, extended to infinitely many degrees of freedom.

## Formalism
The dynamics follow from an action $S[\phi]=\int \mathcal{L}(\phi,\partial_\mu\phi)\,d^dx\,dt$. Extremising $S$ gives the Euler-Lagrange field equations
$$\partial_\mu\frac{\partial\mathcal{L}}{\partial(\partial_\mu\phi)}-\frac{\partial\mathcal{L}}{\partial\phi}=0.$$
- **Examples.** The Klein-Gordon field has $\mathcal{L}=\tfrac12(\partial_t\phi)^2-\tfrac12|\nabla\phi|^2-\tfrac12m^2\phi^2$ and gives a wave equation with mass term; Maxwell's equations follow from the electromagnetic Lagrangian; Ginzburg-Landau theory describes phase transitions by a free-energy functional of an order-parameter field.
- **Noether's theorem for fields** yields conserved currents $\partial_\mu j^\mu=0$ from continuous symmetries ([[conservation_laws]]).
- **Dissipative fields.** Reaction-diffusion and gradient-flow equations $\partial_t\phi=-\delta F/\delta\phi$ do not derive from a conservative action but from a free-energy functional.
- **Boundary and initial data** are needed to close the equations ([[boundary_conditions]]).

**Neural field models** describe population activity $u(x,t)$ on cortical space, for example the Amari equation $\tau\partial_t u=-u+\int w(x-y)f(u(y))\,dy+I$, building on Wilson and Cowan (1973) and Amari (1977). They produce travelling waves, bumps and pattern formation ([[pattern_formation]]).

## Role in Active Inference
The links are with specific research programmes, not with the principle in general. Neural-field and neural-mass equations serve as the generative process in dynamic causal modelling, where model inversion uses variational free energy. Field-theoretic and path-integral methods have been used to write the free energy of continuous systems as an action ([[path_integral_free_energy]]), and Bayesian mechanics casts dynamics as gradient and solenoidal flows on a free-energy landscape, which parallels a field-theory with dissipation. Spatially extended generative models, such as those for image or cortical-sheet inference, treat the hidden state as a field. Claims that the free-energy principle is a field theory of the brain are programmatic and not established.

## Related concepts
- [[quantum_field_theory]] — the quantised counterpart
- [[variational_calculus]] — mathematics of extremising functionals
- [[lagrangian_mechanics]] — particle version of the same action principle
- [[path_integral]] — sums over field or trajectory histories
- [[statistical_physics]] — field theories of collective behaviour
- [[renormalization_group]] — scale-dependence of effective field descriptions

## References
- Wilson, H. R. and Cowan, J. D. (1973). A mathematical theory of the functional dynamics of cortical and thalamic nervous tissue. *Kybernetik*, 13, 55-80.
- Amari, S. (1977). Dynamics of pattern formation in lateral-inhibition type neural fields. *Biological Cybernetics*, 27, 77-87.
