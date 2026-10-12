---
title: Causal Sets
description: Locally finite partially ordered sets proposed as the discrete structure underlying spacetime in an approach to quantum gravity, where the order encodes causality and the number of elements encodes volume.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - quantum_gravity
  - causality
  - order_theory
semantic_relations:
  - type: relates
    links:
      - "[[quantum_gravity]]"
      - "[[minkowski_spacetime]]"
      - "[[lorentzian_manifolds]]"
      - "[[causal_structure]]"
      - "[[graph_theory]]"
      - "[[philosophy_of_physics]]"
---

# Causal Sets

## Overview
Causal set theory is an approach to [[quantum_gravity]] that takes spacetime to be fundamentally discrete. Its basic object, a **causal set** or causet, is a set $C$ of elementary events with a binary relation $\prec$ read as "precedes", which carries the causal structure and nothing else. Bombelli, Lee, Meyer and Sorkin (1987) proposed that the continuum geometry of a [[lorentzian_manifolds|Lorentzian manifold]] is an approximation to such a structure at scales large compared with the discreteness scale.

## Definition
A causal set is a set $C$ with a relation $\prec$ satisfying
1. **Transitivity**: $x\prec y$ and $y\prec z$ imply $x\prec z$.
2. **Irreflexivity (acyclicity)**: $x\not\prec x$, so there are no closed causal loops.
3. **Local finiteness**: for any $x,z\in C$, the interval $\{y: x\prec y\prec z\}$ is finite.

The first two conditions make $C$ a strict partial order, and the third encodes discreteness. Causal sets are often drawn as Hasse diagrams, in which only the covering relations (links) are shown.

## Order plus number equals geometry
Two results about the continuum motivate the slogan. Hawking, King and McCarthy (1976) and Malament (1977) showed that the causal order of a suitably well-behaved spacetime (one that is past and future distinguishing) determines its conformal geometry, that is, the metric up to a local rescaling. What the order cannot fix is the scale factor, and in a causal set this is supplied by counting: the number of elements in a region is taken to be proportional to its spacetime volume. Together, order and number are intended to recover the metric.

## Sprinkling and Lorentz invariance
To compare a causal set with a spacetime, one **sprinkles** points into the manifold at random with a Poisson distribution of density $\rho$, so the expected number in a region of volume $V$ is $\rho V$, and orders them by the causal relation of the manifold. Because the Poisson process picks no preferred direction, a sprinkling into [[minkowski_spacetime]] does not single out a frame, whereas a regular lattice would. This is a central reason the approach is pursued (Surya, 2019).

## Dynamics and geometry from the order
Physical questions are posed in terms of the order alone. Benincasa and Dowker (2010) introduced a family of retarded operators on scalar fields that are defined purely from the order, approximate the continuum d'Alembertian $\Box$ (for slowly varying fields) on a causal set well approximated by four-dimensional Minkowski spacetime, and approximate $\Box-\tfrac12 R$ (with $R$ the Ricci scalar) for sets approximating curved spacetimes. This gives a way to read curvature off the order. Surya (2019) reviews the state of the programme, including its open problems.

## Relation to Active Inference
The relationship is formal and not physical. A causal set is a directed acyclic structure, much like the directed acyclic graphs used for Bayesian networks and [[causal_structure]]. In a causal set, however, the relation records causal precedence between spacetime events, whereas in a graphical model it records statistical dependence between variables. No result in Active Inference depends on causal set theory.

## Related concepts
- [[quantum_gravity]] — the wider setting of attempts to quantise gravity
- [[minkowski_spacetime]] — the flat spacetime into which sprinklings are made
- [[lorentzian_manifolds]] — the continuum models approximated by causal sets
- [[causal_structure]] — causal dependence in graphical models
- [[graph_theory]] — Hasse diagrams and acyclic graphs

## References
- Bombelli, L., Lee, J., Meyer, D., & Sorkin, R. D. (1987). Space-time as a causal set. Physical Review Letters, 59(5), 521-524. https://doi.org/10.1103/PhysRevLett.59.521
- Hawking, S. W., King, A. R., & McCarthy, P. J. (1976). A new topology for curved space-time which incorporates the causal, differential, and conformal structures. Journal of Mathematical Physics, 17(2), 174-181. https://doi.org/10.1063/1.522874
- Malament, D. B. (1977). The class of continuous timelike curves determines the topology of spacetime. Journal of Mathematical Physics, 18(7), 1399-1404. https://doi.org/10.1063/1.523436
- Benincasa, D. M. T., & Dowker, F. (2010). Scalar curvature of a causal set. Physical Review Letters, 104(18), 181301. https://doi.org/10.1103/PhysRevLett.104.181301
- Surya, S. (2019). The causal set approach to quantum gravity. Living Reviews in Relativity, 22, Article 5. https://doi.org/10.1007/s41114-019-0023-1
