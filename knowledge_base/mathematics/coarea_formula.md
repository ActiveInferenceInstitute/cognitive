---
title: Coarea Formula
description: An identity from geometric measure theory that rewrites a volume integral as an integral over the level sets of a Lipschitz map, weighting each slice by the Jacobian, and generalising polar coordinates.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - geometric_measure_theory
  - integration
  - level_sets
semantic_relations:
  - type: relates
    links:
      - "[[measure_theory]]"
      - "[[lebesgue_integration]]"
      - "[[sobolev_spaces]]"
      - "[[differential_geometry]]"
      - "[[partition_function]]"
---

# Coarea Formula

## Overview
The coarea formula expresses an integral over a region of $\mathbb{R}^n$ as an iterated integral: first over the level sets of a function, then over the values the function takes. It is to a general map what Fubini's theorem is to a coordinate projection, with a correction factor that accounts for how fast the map changes. Federer (1959) established it for Lipschitz maps, and Fleming and Rishel (1960) gave the version for functions of bounded variation that links variation to the perimeter of level sets. Standard accounts are Federer (1969) and Evans and Gariepy (1992).

## Statement
Let $f:\mathbb{R}^n\to\mathbb{R}^m$ be Lipschitz with $n\ge m$. Its Jacobian is $Jf(x)=\sqrt{\det\!\big(Df(x)\,Df(x)^{\top}\big)}$, defined almost everywhere. For every integrable $g:\mathbb{R}^n\to\mathbb{R}$,
$$\int_{\mathbb{R}^n} g(x)\,Jf(x)\,dx=\int_{\mathbb{R}^m}\left(\int_{f^{-1}(y)} g\,d\mathcal{H}^{\,n-m}\right)dy,$$
where $\mathcal{H}^{n-m}$ is $(n-m)$-dimensional Hausdorff measure. Taking $g$ to be the indicator of a set $A$ gives $\int_A Jf\,dx=\int_{\mathbb{R}^m}\mathcal{H}^{n-m}(A\cap f^{-1}(y))\,dy$.

For a real-valued function ($m=1$) the Jacobian is $|\nabla f|$, and the formula reads
$$\int_{\mathbb{R}^n} g\,|\nabla f|\,dx=\int_{-\infty}^{\infty}\left(\int_{\{f=t\}} g\,d\mathcal{H}^{\,n-1}\right)dt .$$

## Examples
- **Polar coordinates.** With $f(x)=|x|$ we have $|\nabla f|=1$ away from the origin, and the formula becomes $\int_{\mathbb{R}^n} g\,dx=\int_0^\infty\int_{\partial B_r} g\,d\mathcal{H}^{n-1}\,dr$, the integral over spheres of radius $r$.
- **Volume from slices.** Taking $g=\mathbf 1_{\{a<f<b\}}/|\nabla f|$ on the set where the gradient is nonzero shows that the volume of $\{a<f<b,\ \nabla f\neq0\}$ equals $\int_a^b\int_{\{f=t\}}|\nabla f|^{-1}\,d\mathcal{H}^{n-1}\,dt$.

## Consequences
- **Total variation and perimeter.** For a function $u$ of bounded variation, the total variation equals the integral over $t$ of the perimeter of the superlevel set $\{u>t\}$. This is the Fleming-Rishel formula.
- **Isoperimetric and Sobolev inequalities.** Combining it with the isoperimetric inequality yields Sobolev inequalities for functions of bounded variation (see [[sobolev_spaces]]).
- **Smooth case.** If $f$ is smooth, almost every value $y$ is regular (Sard's theorem), and the fibre $f^{-1}(y)$ is an embedded submanifold whose surface measure is $\mathcal{H}^{n-m}$.

## Relation to Active Inference
The connection is interpretive. Many quantities in the framework are integrals over a state space of a function of an energy-like scalar, such as free energy or surprisal. The coarea formula is the standard way to decompose such integrals by level set, in the same way that the microcanonical density of states of statistical mechanics is the surface integral of $1/|\nabla H|$ over an energy shell (see [[partition_function]]). The formula is a general analytic tool, and Active Inference does not depend on it.

## Related concepts
- [[measure_theory]] — Hausdorff and Lebesgue measures
- [[lebesgue_integration]] — the integral being decomposed
- [[sobolev_spaces]] — function spaces whose inequalities use the formula
- [[differential_geometry]] — level sets as submanifolds
- [[partition_function]] — the density of states as a slice integral

## References
- Federer, H. (1959). Curvature measures. Transactions of the American Mathematical Society, 93(3), 418-491. https://doi.org/10.1090/S0002-9947-1959-0110078-1
- Fleming, W. H., & Rishel, R. (1960). An integral formula for total gradient variation. Archiv der Mathematik, 11(1), 218-222. https://doi.org/10.1007/BF01236935
- Federer, H. (1969). Geometric Measure Theory. Springer.
- Evans, L. C., & Gariepy, R. F. (1992). Measure Theory and Fine Properties of Functions. CRC Press.
