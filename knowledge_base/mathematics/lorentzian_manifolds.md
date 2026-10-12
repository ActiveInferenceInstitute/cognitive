---
title: Lorentzian Manifolds
description: Smooth manifolds equipped with a metric of Lorentzian signature, the mathematical model of spacetime in general relativity, with causal structure, curvature and the Einstein field equations.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - general_relativity
  - spacetime
  - differential_geometry
semantic_relations:
  - type: relates
    links:
      - "[[minkowski_spacetime]]"
      - "[[differential_geometry]]"
      - "[[metric_tensor]]"
      - "[[black_hole_thermodynamics]]"
      - "[[causal_sets]]"
      - "[[information_geometry]]"
---

# Lorentzian Manifolds

## Overview
A Lorentzian manifold is a smooth manifold $M$ with a smooth, nondegenerate symmetric metric $g$ whose signature is $(-,+,\dots,+)$: one negative direction and the rest positive (some authors use the opposite sign convention). General relativity (Einstein, 1916) models spacetime as a four-dimensional Lorentzian manifold, with the metric encoding both the causal structure and the gravitational field. It is the curved counterpart of [[minkowski_spacetime]], which is the case in which $g$ is constant. The standard mathematical treatments are O'Neill (1983), Hawking and Ellis (1973) and Wald (1984).

## Causal character and time orientation
Because $g$ is indefinite, each tangent vector $v$ is timelike ($g(v,v)<0$), null ($g(v,v)=0$) or spacelike ($g(v,v)>0$). The nonzero null vectors at a point form a double cone. Choosing one half-cone consistently across the manifold, when possible, gives a time orientation. A curve is causal if its tangent vectors are timelike or null. The chronological future $I^+(p)$ is the set of points reachable from $p$ by a future-directed timelike curve, and $J^+(p)$ is the corresponding set for causal curves. A spacetime is **globally hyperbolic** if it has a Cauchy surface, a hypersurface met exactly once by every inextendible causal curve; for such spacetimes the initial value problem of the field equations is well posed.

## Curvature and field equations
The Levi-Civita connection of $g$ defines the Riemann curvature tensor, and contractions give the Ricci tensor $R_{\mu\nu}$ and scalar curvature $R$. Einstein's equations relate the curvature to the energy and momentum content of matter:
$$G_{\mu\nu}+\Lambda g_{\mu\nu}=\frac{8\pi G}{c^4}\,T_{\mu\nu},\qquad G_{\mu\nu}=R_{\mu\nu}-\tfrac12 R\,g_{\mu\nu},$$
where $T_{\mu\nu}$ is the stress-energy tensor and $\Lambda$ the cosmological constant. Freely falling observers follow timelike geodesics, and in suitable coordinates around any point the metric agrees with the Minkowski metric to first order, which is the mathematical form of the equivalence principle.

## Standard examples
- **Schwarzschild spacetime**, the spherically symmetric vacuum solution: $ds^2=-(1-r_s/r)c^2dt^2+(1-r_s/r)^{-1}dr^2+r^2d\Omega^2$ with $r_s=2GM/c^2$. The surface $r=r_s$ is an event horizon (see [[black_hole_thermodynamics]]).
- **Friedmann-Lemaître-Robertson-Walker spacetimes**: $ds^2=-c^2dt^2+a(t)^2\left[\frac{dr^2}{1-kr^2}+r^2d\Omega^2\right]$, the homogeneous and isotropic models used in cosmology.

## Singularities
Penrose (1965) showed that a spacetime with a noncompact Cauchy surface that satisfies the null energy condition and contains a closed trapped surface must be null geodesically incomplete. Theorems of this type (see also Hawking and Ellis, 1973) do not assume exact symmetry, which is why geodesic incompleteness is regarded as a generic feature of the theory rather than an artefact of special solutions. The Stanford Encyclopedia entry on singularities and black holes discusses how the notion of singularity is itself subtle.

## Relation to Active Inference
There is no established connection. The geometry that appears in Active Inference research is mainly Riemannian: the Fisher information metric on a statistical manifold is positive definite (see [[information_geometry]]), whereas a Lorentzian metric is indefinite and is tied to physical causal structure. Analogies between the two are speculation and should not be presented as results.

## Related concepts
- [[minkowski_spacetime]] — the flat special case
- [[differential_geometry]] — manifolds, connections and curvature
- [[metric_tensor]] — the general notion of a metric
- [[black_hole_thermodynamics]] — horizons and their thermodynamics
- [[causal_sets]] — a discrete approach to quantum gravity

## References
- Einstein, A. (1916). Die Grundlage der allgemeinen Relativitätstheorie. Annalen der Physik, 354(7), 769-822. https://doi.org/10.1002/andp.19163540702
- Penrose, R. (1965). Gravitational collapse and space-time singularities. Physical Review Letters, 14(3), 57-59. https://doi.org/10.1103/PhysRevLett.14.57
- Hawking, S. W., & Ellis, G. F. R. (1973). The Large Scale Structure of Space-Time. Cambridge University Press.
- O'Neill, B. (1983). Semi-Riemannian Geometry with Applications to Relativity. Academic Press.
- Wald, R. M. (1984). General Relativity. University of Chicago Press.
- Curiel, E., Visser, M., & Doboszewski, J. Singularities and Black Holes. Stanford Encyclopedia of Philosophy. https://plato.stanford.edu/entries/spacetime-singularities/
