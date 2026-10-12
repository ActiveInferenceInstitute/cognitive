---
title: Minkowski Spacetime
description: The flat four-dimensional geometry of special relativity, a vector space with an indefinite metric whose invariant interval defines light cones, proper time and the causal structure of events.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - relativity
  - spacetime
  - geometry
semantic_relations:
  - type: relates
    links:
      - "[[lorentzian_manifolds]]"
      - "[[causal_sets]]"
      - "[[metric_tensor]]"
      - "[[differential_geometry]]"
      - "[[lie_groups]]"
      - "[[philosophy_of_time]]"
---

# Minkowski Spacetime

## Overview
Minkowski spacetime is the geometric setting of special relativity. Einstein (1905) derived the theory from two postulates, the equivalence of inertial frames and the constancy of the speed of light. Minkowski then recast it as the geometry of a four-dimensional space whose points are events, so that the relativistic effects become properties of one invariant structure instead of frame-dependent corrections. It is the simplest example of a Lorentzian geometry and the flat model for the curved spacetimes of general relativity (see [[lorentzian_manifolds]]).

## Definition
Minkowski spacetime is $\mathbb{R}^4$ equipped with the bilinear form
$$\eta(x,y) = -x^0y^0 + x^1y^1 + x^2y^2 + x^3y^3,$$
in units with the speed of light equal to $1$. The form is nondegenerate but indefinite, with signature $(-,+,+,+)$ (many authors use the opposite sign). The invariant interval between events separated by $\Delta x$ is $\eta(\Delta x,\Delta x)$. A vector $v$ is **timelike** if $\eta(v,v)<0$, **null** if $\eta(v,v)=0$ and **spacelike** if $\eta(v,v)>0$.

## Symmetries and invariants
The linear maps preserving $\eta$ form the Lorentz group $O(1,3)$, a [[lie_groups|Lie group]] of dimension $6$. Adding translations gives the Poincaré group, of dimension $10$, which is the symmetry group of the theory. Lorentz boosts mix the time and space coordinates, which produces the relativity of simultaneity, time dilation and length contraction. These are consequences of keeping $\eta$ fixed, not additional assumptions.

## Causal structure and proper time
At each event the null vectors form a double cone, the light cone, separating the events that can influence it from those it can influence. Event $q$ lies in the causal future of $p$ if $q-p$ is a future-directed timelike or null vector. Zeeman (1964) proved that a bijection of Minkowski space preserving this causal order must be a Poincaré transformation combined with a dilation, so the causal order alone determines the geometry up to scale.

For a timelike worldline, the proper time is $\tau=\int\sqrt{-\eta(\dot x,\dot x)}\,d\lambda$, the time recorded by a clock moving along it. Among timelike curves joining two events, a straight inertial path has the longest proper time, which is the geometric content of the twin effect.

## Role in later theories
Gravity is described by letting the metric vary from event to event, with Minkowski spacetime as the local approximation at each point (see [[lorentzian_manifolds]]). Approaches that start from the causal order, such as [[causal_sets]], treat the cone structure as primary and ask how the metric emerges from it. Standard textbooks include Naber (1992) for the mathematics and the Stanford Encyclopedia entry on inertial frames for the conceptual background.

## Relation to Active Inference
There is no established connection. Standard Active Inference models treat time as a parameter indexing a sequence of states, not as part of a Lorentzian geometry, and the framework makes no relativistic assumptions. The page matters here mainly as the background for geometric and ontological discussions of space and time (see [[philosophy_of_time]]).

## Related concepts
- [[lorentzian_manifolds]] — curved generalisations with the same signature
- [[causal_sets]] — a discrete theory built on the causal order
- [[metric_tensor]] — the general notion of a metric, here indefinite
- [[lie_groups]] — the Lorentz and Poincaré groups
- [[philosophy_of_time]] — conceptual issues about time and simultaneity

## References
- Einstein, A. (1905). Zur Elektrodynamik bewegter Körper. Annalen der Physik, 322(10), 891-921. https://doi.org/10.1002/andp.19053221004
- Zeeman, E. C. (1964). Causality implies the Lorentz group. Journal of Mathematical Physics, 5(4), 490-493. https://doi.org/10.1063/1.1704140
- Naber, G. L. (1992). The Geometry of Minkowski Spacetime: An Introduction to the Mathematics of the Special Theory of Relativity. Springer.
- DiSalle, R. Space and Time: Inertial Frames. Stanford Encyclopedia of Philosophy. https://plato.stanford.edu/entries/spacetime-iframes/
