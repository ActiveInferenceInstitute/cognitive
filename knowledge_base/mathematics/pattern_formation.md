---
title: Pattern Formation
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - pattern_formation
  - reaction_diffusion
  - turing
  - morphogenesis
semantic_relations:
  - type: foundation
    links:
      - "[[differential_equations]]"
      - "[[dynamical_systems]]"
  - type: relates
    links:
      - "[[self_organization]]"
      - "[[synergetics]]"
      - "[[emergence]]"
---

# Pattern Formation

## Overview
Pattern formation is the spontaneous appearance of spatial or spatio-temporal structure in a system that starts out nearly homogeneous. The canonical mechanism is a symmetry-breaking instability: a uniform state becomes unstable to perturbations of a particular wavelength, and that mode grows into a stable pattern.

## Definition / Formalism
Reaction-diffusion systems have the form $\partial_t u=f(u,v)+D_u\nabla^2u$, $\partial_t v=g(u,v)+D_v\nabla^2v$. Let $J=\begin{pmatrix}f_u&f_v\\g_u&g_v\end{pmatrix}$ be the Jacobian at a homogeneous steady state. Without diffusion the state is stable when $\operatorname{tr}J<0$ and $\det J>0$. A diffusion-driven (Turing) instability, discovered by Turing in 1952, occurs for spatial perturbations $e^{ikx}$ when, in addition,

$$D_vf_u+D_ug_v>0,\qquad (D_vf_u+D_ug_v)^2>4D_uD_v\det J.$$

The unstable band of $k^2$ lies between the roots of $D_uD_vk^4-(D_vf_u+D_ug_v)k^2+\det J=0$. Example: $J=\begin{pmatrix}0.4&-0.6\\0.8&-0.8\end{pmatrix}$ (trace $-0.4$, determinant $0.16$), $D_u=1$, $D_v=20$ satisfies both conditions ($7.2>0$ and $51.8>12.8$); the largest growth rate is negative at $k=0$ and positive, about $0.137$, near $k\approx0.33$. The requirement of a much faster-diffusing inhibitor is the local-activation, long-range-inhibition principle. Other mechanisms include Rayleigh-Benard convection, the Swift-Hohenberg equation, and Haken's synergetics, in which fast variables are slaved to a few slowly evolving order parameters near an instability.

## Role in Active Inference
Morphogenesis has been treated as inference by Friston, Levin, Sengupta and Pezzulo (2015): cells that minimise free energy under a shared generative model of the target form can regulate a pattern from local signals. That model is a proof of principle showing that pattern regulation can be cast as inference; it does not demonstrate that real tissue implements those equations. More generally, a pattern is a low-entropy, non-equilibrium steady state that requires continuing dissipation, and the free energy principle supplies a statistical description of such states. The Turing analysis above is classical dynamical-systems theory and does not depend on the free energy principle.

## Related concepts
- [[self_organization]] — formation of order from local interactions
- [[synergetics]] — order parameters and the slaving principle
- [[emergence]] — macroscopic structure not present at the component level
- [[synchronization]] — temporal analogue of spatial ordering
- [[non_equilibrium_steady_state]] — the statistical setting for sustained patterns
- [[phase_transitions]] — symmetry breaking as a transition

## References
- Turing (1952), The chemical basis of morphogenesis, Philosophical Transactions of the Royal Society B 237.
- Haken (1983), Synergetics: An Introduction.
- Friston, Levin, Sengupta & Pezzulo (2015), Knowing one's place: a free-energy approach to pattern regulation, Journal of the Royal Society Interface 12.
