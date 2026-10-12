---
title: Normal Forms
description: Simplest representatives of a dynamical system near an equilibrium, obtained by near-identity coordinate changes, and the canonical normal forms of the elementary bifurcations.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - bifurcation
  - local_analysis
semantic_relations:
  - type: relates
    links:
      - "[[bifurcation_theory]]"
      - "[[center_manifold]]"
      - "[[hopf_bifurcation]]"
      - "[[saddle_node_bifurcation]]"
      - "[[structural_stability]]"
---

# Normal Forms

## Overview
In dynamical systems, a normal form is a simplified expression of a vector field near an equilibrium, obtained by a smooth change of coordinates that removes as many nonlinear terms as possible. The aim is to retain only the terms that cannot be eliminated and that therefore determine the local qualitative dynamics. Normal form theory goes back to Poincare and Dulac.

## Poincare-Dulac theory
Consider $\dot x=Ax+f(x)$ with $f$ starting at quadratic order, and suppose $A$ is diagonalisable with eigenvalues $\lambda_1,\dots,\lambda_n$. A near-identity transformation $x=y+h(y)$, with $h$ polynomial of order $k$, can remove a monomial $y_1^{m_1}\cdots y_n^{m_n}$ from the equation for $y_j$ unless the resonance condition
$$\lambda_j=\sum_i m_i\lambda_i,\qquad \sum_i m_i=k,$$
holds. The terms that satisfy it are called resonant and remain in the normal form. The transformation is generally a formal power series, and convergence is not guaranteed.

## Normal forms of elementary bifurcations
Normal forms are most useful at bifurcation points, after reduction to a [[center_manifold]]. For a one-dimensional system with parameter $\mu$:
- saddle-node: $\dot x=\mu+x^2$ (see [[saddle_node_bifurcation]]);
- transcritical: $\dot x=\mu x-x^2$;
- pitchfork: $\dot x=\mu x\mp x^3$, with the sign selecting the supercritical or subcritical case.

For a pair of complex eigenvalues crossing the imaginary axis, the Hopf normal form in complex notation is
$$\dot z=(\mu+i\omega)z+c\,z|z|^2,$$
where the sign of $\operatorname{Re}c$ determines whether the bifurcating limit cycle is stable (supercritical) or unstable ([[hopf_bifurcation]]). Because every system with the same local structure can be mapped to one of these forms, they serve as universal unfoldings: quantitative features such as the square-root scaling of the amplitude near a supercritical Hopf point follow from the form alone.

## Relation to Active Inference
The link is indirect. The [[free_energy_principle]] is formulated in terms of flows on a state space, and analyses of the stability of such flows, for example of critical slowing near a bifurcation, use the local tools described here. The use of normal forms in neural mass and population models is an established area of computational neuroscience, but normal form theory is not itself part of the Active Inference formalism, and applications to specific generative models remain case-by-case.

## Related concepts
- [[bifurcation_theory]] — the theory whose local models normal forms provide
- [[center_manifold]] — reduction that precedes normal form computation
- [[hopf_bifurcation]] — onset of oscillation, with its complex normal form
- [[saddle_node_bifurcation]] — creation or annihilation of equilibria
- [[structural_stability]] — robustness of qualitative dynamics under perturbation
- [[dynamical_systems]] — the general setting

## References
- Guckenheimer, J., & Holmes, P. (1983). Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields. Springer.
- Kuznetsov, Y. A. (2004). Elements of Applied Bifurcation Theory (3rd ed.). Springer.
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
