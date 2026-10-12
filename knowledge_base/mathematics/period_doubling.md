---
title: Period Doubling
description: The bifurcation sequence in which a periodic orbit loses stability to one of twice the period, its universal Feigenbaum scaling, and its role as a route to chaos.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - bifurcation
  - chaos
semantic_relations:
  - type: relates
    links:
      - "[[bifurcation_theory]]"
      - "[[bifurcation_diagrams]]"
      - "[[chaos_theory]]"
      - "[[lyapunov_exponents]]"
      - "[[nonlinearity]]"
---

# Period Doubling

## Overview
A period-doubling (flip) bifurcation occurs when a periodic orbit of period $T$ becomes unstable as a parameter is varied and a new stable orbit of period $2T$ appears nearby. For a one-dimensional map the condition is that the multiplier of the fixed point passes through $-1$. Repeated period doubling can produce an infinite cascade that accumulates at a finite parameter value, beyond which chaotic behaviour appears. This is one of the standard routes from regular dynamics to chaos.

## The logistic map
The logistic map $x_{n+1}=r\,x_n(1-x_n)$ on $[0,1]$ shows the cascade clearly (May, 1976). The non-zero fixed point $x^*=1-1/r$ is stable for $1<r<3$, and at $r=3$ its multiplier $f'(x^*)=2-r$ reaches $-1$. A stable 2-cycle then appears, which doubles to period 4 at $r=1+\sqrt6\approx3.449$. The cascade accumulates at $r_\infty\approx3.5699$, followed by a regime that mixes chaotic bands with periodic windows, such as the period-3 window beginning at $r=1+\sqrt8\approx3.828$ ([[bifurcation_diagrams]]).

## Feigenbaum universality
If $r_n$ is the parameter value of the $n$-th doubling, the ratios
$$\delta=\lim_{n\to\infty}\frac{r_n-r_{n-1}}{r_{n+1}-r_n}\approx4.6692$$
converge to a constant. Feigenbaum (1978) found numerically that this constant is the same for a wide class of unimodal maps with a quadratic maximum, and a second constant $\alpha\approx2.5029$ governs the scaling of the orbit's spatial structure. The universality was later explained by a renormalisation-group argument for functional equations ([[renormalization_group]]). The values quoted are numerical constants, and the universality applies to maps with a quadratic extremum. A different order of the maximum gives different constants.

Period doubling also appears in flows, for example in forced oscillators and in some fluid and chemical systems, where it is detected by following Floquet multipliers.

## Relation to Active Inference
There is no direct relationship between period doubling and the Active Inference formalism. It is relevant to the vault as part of the toolkit for analysing nonlinear neural and population models that might be used as generative processes, and for the general question of how a parameter change alters the qualitative character of dynamics ([[nonlinearity]], [[free_energy_principle]]). Claims that cognitive systems operate near such cascades should be treated as hypotheses about particular models.

## Related concepts
- [[bifurcation_theory]] — the framework for qualitative change in parameters
- [[bifurcation_diagrams]] — the standard display of the cascade
- [[chaos_theory]] — the regime reached after accumulation
- [[lyapunov_exponents]] — quantify the onset of chaos
- [[hopf_bifurcation]] — another route by which oscillation arises
- [[renormalization_group]] — the explanation of Feigenbaum universality

## References
- May, R. M. (1976). Simple mathematical models with very complicated dynamics. Nature, 261, 459-467.
- Feigenbaum, M. J. (1978). Quantitative universality for a class of nonlinear transformations. Journal of Statistical Physics, 19, 25-52.
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
