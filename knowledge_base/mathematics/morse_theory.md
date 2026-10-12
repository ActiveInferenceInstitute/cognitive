---
title: Morse Theory
description: The study of how the critical points of a smooth function on a manifold determine the manifold's topology, through sublevel sets, indices and the Morse inequalities.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - differential_topology
  - critical_points
  - manifolds
semantic_relations:
  - type: foundation
    links:
      - "[[topology]]"
      - "[[differential_geometry]]"
      - "[[calculus]]"
  - type: relates
    links:
      - "[[euler_characteristic]]"
      - "[[algebraic_topology]]"
      - "[[gradient_flows]]"
      - "[[dynamical_systems]]"
      - "[[basin_of_attraction]]"
---

# Morse Theory

## Overview
Morse theory relates the shape of a smooth manifold $M$ to the critical points of a smooth real function $f:M\to\mathbb{R}$. Marston Morse developed it to study geodesics through the calculus of variations in the large (Morse, 1934), and Milnor (1963) gave it the form in which it is now usually taught. Its central message is that the topology of $M$ can be read from where $f$ has critical points and what kind they are.

## Basic notions
A point $p$ is critical if $df_p=0$. It is nondegenerate if the Hessian matrix of second derivatives at $p$ is invertible, and $f$ is a Morse function if every critical point is nondegenerate. The index of a critical point is the number of negative eigenvalues of the Hessian. The Morse lemma says that near a nondegenerate critical point of index $\lambda$ there are local coordinates in which

$$f(x)=f(p)-x_1^2-\cdots-x_\lambda^2+x_{\lambda+1}^2+\cdots+x_n^2 .$$

Minima have index $0$, maxima have index $n$, and saddles lie between. Morse functions are generic: a small perturbation turns almost any smooth function into one.

## Sublevel sets and cells
Consider the sublevel sets $M^a=\{x: f(x)\le a\}$. As $a$ increases, the topology of $M^a$ does not change while $a$ passes no critical value. When $a$ passes a nondegenerate critical point of index $\lambda$, the homotopy type of $M^a$ changes by attaching a $\lambda$-dimensional cell (Milnor, 1963). For compact $M$, this builds $M$ up to homotopy equivalence as a cell complex with one cell of dimension $\lambda$ for each critical point of index $\lambda$.

## Morse inequalities
Let $c_k$ be the number of critical points of index $k$ and $b_k$ the $k$-th Betti number of $M$. Then

$$c_k\ge b_k,\qquad \sum_k(-1)^k c_k=\sum_k(-1)^k b_k=\chi(M),$$

so a Morse function on $M$ must have at least as many critical points as the topology demands, and the alternating count gives the Euler characteristic (see [[euler_characteristic]]). A sphere, for example, always has at least a minimum and a maximum. Bott (1982) surveys the theory and its later development.

## Relation to Active Inference
The connection is interpretive. Free energy $F$ or a loss function can be regarded as a function on a space of beliefs or parameters, and its critical points (minima, saddles, maxima) organise the behaviour of gradient flows that minimise it. Morse theory is the mathematical theory of that organisation: the number and index of critical points constrain the topology of the parameter space, and the unstable manifolds of a gradient flow decompose the space into cells. Standard Active Inference practice does not use Morse theory, and no result in the Active Inference literature depends on it. Real free-energy landscapes also often have degenerate critical points, to which the nondegeneracy hypothesis does not apply.

## Related concepts
- [[euler_characteristic]] — the alternating count of critical points
- [[topology]] — the invariants Morse theory recovers
- [[gradient_flows]] — the flows whose structure Morse functions organise
- [[basin_of_attraction]] — stable manifolds of minima
- [[dynamical_systems]] — gradient systems and their critical points

## References
- Milnor, J. (1963). Morse Theory (Annals of Mathematics Studies, No. 51). Princeton University Press.
- Morse, M. (1934). The Calculus of Variations in the Large (American Mathematical Society Colloquium Publications). American Mathematical Society.
- Bott, R. (1982). Lectures on Morse theory, old and new. Bulletin of the American Mathematical Society, 7(2), 331-358. https://doi.org/10.1090/S0273-0979-1982-15038-8
