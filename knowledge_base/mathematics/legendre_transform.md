---
title: Legendre Transform
description: The convex-conjugate transformation that exchanges a function of one variable for a function of its slope, with its use in mechanics, thermodynamics and the duality behind variational inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - convex_analysis
  - thermodynamics
  - duality
semantic_relations:
  - type: relates
    links:
      - "[[convex_optimization]]"
      - "[[hamiltonian_mechanics]]"
      - "[[thermodynamics]]"
      - "[[exponential_family]]"
      - "[[variational_free_energy]]"
---

# Legendre Transform

## Overview
The Legendre transform replaces a function $f(x)$ by a function of its slope $p=f'(x)$, keeping all the information in $f$ when $f$ is convex. In the general convex setting it is called the Legendre-Fenchel transform or convex conjugate,
$$f^*(p)=\sup_{x}\,\bigl(px-f(x)\bigr).$$
For a differentiable, strictly convex $f$ the supremum is attained where $f'(x)=p$, so $f^*(p)=px-f(x)$ evaluated at that point, and $(f^*)'(p)=x$. The slope and the variable exchange roles.

## Properties and examples
- **Involution.** If $f$ is convex and lower semicontinuous, then $f^{**}=f$.
- **Fenchel-Young inequality.** $f(x)+f^*(p)\ge px$, with equality when $p$ is a subgradient of $f$ at $x$.
- **Examples.** The conjugate of $\tfrac12x^2$ is $\tfrac12p^2$. The conjugate of $e^{x}$ is $p\ln p-p$ for $p>0$. The conjugate of $\tfrac12 x^\top Ax$ with $A$ positive definite is $\tfrac12 p^\top A^{-1}p$.
- **Convexity.** The transform $f^*$ is convex even when $f$ is not.

## Uses in physics
In mechanics the Hamiltonian is the Legendre transform of the Lagrangian with respect to velocity, $H(q,p)=p\dot q-L(q,\dot q)$ with $p=\partial L/\partial\dot q$. In thermodynamics the internal energy $U(S,V)$ is converted into the Helmholtz free energy $F=U-TS$ by trading entropy for temperature, and into the Gibbs free energy $G=F+PV$ by trading volume for pressure.

## Relation to Active Inference
The link is a mathematical one that underlies variational inference. For an exponential family the log-partition function $A(\theta)$ is convex and its conjugate is the negative entropy of the distribution with the matching mean parameters (Wainwright and Jordan, 2008). Writing $\ln Z$ as a supremum over distributions of an expected log-density plus entropy gives the Gibbs variational principle, from which the free energy bound used in Active Inference follows. In this sense variational free energy is a Legendre-type dual of the log evidence. Whether this duality has a further role in the theory is a matter of interpretation.

## Related concepts
- [[convex_optimization]] — the setting in which conjugates are defined and used
- [[hamiltonian_mechanics]] — built from the Lagrangian by this transformation
- [[thermodynamics]] — where free energies arise as transforms of internal energy
- [[exponential_family]] — log-partition function and entropy as a conjugate pair
- [[variational_free_energy]] — the bound obtained from this duality
- [[partition_function]] — whose logarithm is the function being transformed

## References
- Rockafellar, R. T. (1970). Convex Analysis. Princeton University Press.
- Zia, R. K. P., Redish, E. F., & McKay, S. R. (2009). Making sense of the Legendre transform. American Journal of Physics, 77, 614-622.
- Wainwright, M. J., & Jordan, M. I. (2008). Graphical models, exponential families, and variational inference. Foundations and Trends in Machine Learning, 1, 1-305.
