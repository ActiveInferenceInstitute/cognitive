---
title: Center Manifold
description: The invariant manifold tangent to the neutral eigendirections of an equilibrium, the center manifold theorem and reduction principle, a worked example, and links to timescale separation in the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - bifurcation
  - dimension_reduction
semantic_relations:
  - type: relates
    links:
      - "[[bifurcation_theory]]"
      - "[[normal_forms]]"
      - "[[linear_stability]]"
      - "[[timescale_separation]]"
      - "[[dynamical_systems]]"
      - "[[synergetics]]"
wikidata: Q4504202
---

# Center Manifold

## Overview
The center manifold is a tool for analysing a nonlinear dynamical system near an equilibrium whose linearisation is not decisive. For $\dot{z} = F(z)$ with $F(0) = 0$, the eigenvalues of the Jacobian $DF(0)$ are divided into stable (negative real part), unstable (positive real part) and central (zero real part). When some have zero real part, linear analysis cannot determine stability, and the behaviour near the equilibrium is governed by the dynamics along the central directions. The center manifold is an invariant manifold tangent to the central eigenspace at the equilibrium.

## Theorem and reduction
Write the system in coordinates as
$$\dot{x} = Ax + f(x, y), \qquad \dot{y} = By + g(x, y),$$
where $A$ has all eigenvalues with zero real part, $B$ has all eigenvalues with nonzero real part, and $f, g$ vanish together with their first derivatives at the origin. The center manifold theorem states that there is a locally defined invariant manifold $y = h(x)$ with $h(0) = 0$ and $Dh(0) = 0$, as smooth as the vector field (Carr, 1981). It is generally not unique, but all choices share one Taylor expansion. Invariance requires
$$Dh(x)\,[Ax + f(x, h(x))] = B\,h(x) + g(x, h(x)),$$
which is solved order by order as a power series. The reduction principle then states that the local behaviour of the full system is determined by the reduced equation $\dot{x} = Ax + f(x, h(x))$ on the manifold, together with the hyperbolic behaviour in the remaining directions.

## Example
Consider $\dot{x} = xy$, $\dot{y} = -y - x^2$. Here $A = 0$ and $B = -1$. Substituting $h(x) = a x^2 + \dots$ into the invariance equation gives $-a x^2 - x^2$ on the right and terms of order $x^4$ on the left, so $a = -1$ and $h(x) = -x^2 + O(x^4)$. The reduced equation is $\dot{x} = -x^3 + O(x^5)$, so the origin is asymptotically stable, which the linearisation cannot show.

## Relation to Active Inference
The center manifold theorem is related to the idea that fast variables are slaved to slow ones, which was developed in synergetics, where slowly changing order parameters constrain many fast modes (see [[synergetics]]). Work on the free energy principle uses a similar separation of timescales. The correspondence is an analogy at the level of method: center manifold reduction applies to a deterministic system near a particular equilibrium, and the free energy principle is stated for stochastic systems at a steady state.

## Related concepts
- [[bifurcation_theory]] — the main application of center manifold reduction
- [[normal_forms]] — simplified forms obtained on the manifold
- [[linear_stability]] — the analysis that fails at central eigenvalues
- [[timescale_separation]] — the slow-fast structure
- [[hopf_bifurcation]] — a case with a two-dimensional center manifold
- [[synergetics]] — the slaving principle

## References
- Carr, J. (1981). Applications of Centre Manifold Theory. Springer.
- Guckenheimer, J., & Holmes, P. (1983). Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields. Springer.
- Kuznetsov, Y. A. (2004). Elements of Applied Bifurcation Theory (3rd ed.). Springer.
