---
title: Geometric Phase
description: The geometric phase is a phase shift acquired by a system during cyclic evolution that depends on the path traced in parameter space and not on elapsed time, as in the Berry phase.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - quantum_mechanics
  - differential_geometry
  - holonomy
semantic_relations:
  - type: relates
    links:
      - "[[berry_connection]]"
      - "[[holonomy]]"
      - "[[fiber_bundles]]"
      - "[[quantum_mechanics]]"
      - "[[geometric_mechanics]]"
wikidata: Q3283
---

# Geometric Phase

## Overview
When a quantum system is carried around a closed loop in a space of parameters, its state returns to itself up to a phase. Part of this phase is the ordinary dynamical phase, which grows with time and energy. The remainder depends only on the geometry of the loop, and is called the geometric phase. Berry showed in 1984 that for adiabatic cyclic evolution this extra phase is generally present and physically meaningful, and Simon identified it as a holonomy.

## Berry phase
Let $H(\mathbf{R})$ depend on parameters $\mathbf{R}$ and let $|n(\mathbf{R})\rangle$ be a nondegenerate eigenstate. If $\mathbf{R}$ is changed slowly around a closed loop $C$, the state acquires the geometric phase
$$\gamma_n(C)=i\oint_C\langle n(\mathbf{R})|\nabla_{\mathbf{R}}\,n(\mathbf{R})\rangle\cdot d\mathbf{R}=\oint_C\mathbf{A}_n\cdot d\mathbf{R},$$
where $\mathbf{A}_n$ is the Berry connection. A change of the arbitrary phase of $|n\rangle$ alters $\mathbf{A}_n$ by a gradient, so $\gamma_n$ is gauge invariant modulo $2\pi$. By Stokes' theorem, the phase equals the flux of the Berry curvature $\nabla\times\mathbf{A}_n$ through any surface bounded by $C$. For a spin-$\tfrac12$ particle in a slowly rotated magnetic field the phase is $\mp\tfrac12\Omega$ for the two spin states, where $\Omega$ is the solid angle swept out by the field direction.

## Generalisations and analogues
Aharonov and Anandan removed the adiabatic restriction, and Pancharatnam described an earlier version in the context of the polarisation of light. Classical analogues include Hannay's angle, the shift in angle variables for adiabatically varied integrable systems, and the rotation of the Foucault pendulum's plane of oscillation, which is a holonomy of parallel transport on a sphere. The Aharonov-Bohm effect is another phase determined by a loop integral of a gauge potential. All are expressed as the holonomy of a connection on a fibre bundle.

## Relation to Active Inference
There is no established role for geometric phases in Active Inference. They appear in the vault because path-integral formulations use the same differential-geometric language, with the term $\oint A_\mu\,dx^\mu$ describing path-dependent phase contributions in the action. Any application to inference dynamics is formal or exploratory, and does not depend on quantum behaviour of neural systems.

## Related concepts
- [[berry_connection]] — the connection whose loop integral is the phase
- [[holonomy]] — the general notion of change after parallel transport around a loop
- [[fiber_bundles]] — the mathematical home of the connection
- [[quantum_mechanics]] — the setting where adiabatic phases were identified
- [[geometric_mechanics]] — classical analogues such as Hannay's angle
- [[path_integral_theory]] — formulations in which phase factors enter the action

## References
- Berry, M. V. (1984). Quantal phase factors accompanying adiabatic changes. Proceedings of the Royal Society of London A, 392, 45-57.
- Simon, B. (1983). Holonomy, the quantum adiabatic theorem, and Berry's phase. Physical Review Letters, 51, 2167-2170.
- Shapere, A., & Wilczek, F. (Eds.). (1989). Geometric Phases in Physics. World Scientific.
