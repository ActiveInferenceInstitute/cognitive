---
title: Holographic Principle
description: The holographic principle proposes that the information content of a region of space is bounded by the area of its boundary, motivated by black hole entropy and realised in the AdS/CFT correspondence.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - theoretical_physics
  - information_theory
  - black_holes
semantic_relations:
  - type: relates
    links:
      - "[[black_hole_thermodynamics]]"
      - "[[quantum_information]]"
      - "[[entropy]]"
      - "[[markov_blanket]]"
      - "[[quantum_free_energy]]"
---

# Holographic Principle

## Overview
The holographic principle is a conjecture in theoretical physics that the physics of a region of space, including gravity, can be described by degrees of freedom living on its boundary, with about one unit of information per Planck area. It was formulated by 't Hooft and given a string-theoretic interpretation by Susskind, and it is regarded as a guide to quantum gravity, not as a derived or experimentally tested theorem.

## Origin in black hole entropy
Bekenstein proposed that a black hole has entropy proportional to the area of its event horizon, and Hawking's calculation of black hole radiation fixed the coefficient. In units with $\hbar=c=k_B=1$ the entropy is
$$S_{\mathrm{BH}}=\frac{A}{4G},$$
where $A$ is the horizon area and $G$ is Newton's constant. This is surprising because the entropy of ordinary systems scales with volume. A thought experiment shows that no region of space can contain more entropy than a black hole that would fit inside it. This suggests the bound
$$S\le\frac{A}{4G}$$
for the entropy in a region with boundary area $A$, which Bousso generalised to a covariant bound, stated for the entropy passing through light-sheets of the boundary in general spacetimes. The number of independent states is thus much smaller than a local field theory in the bulk would suggest.

## AdS/CFT correspondence
The best-understood realisation is Maldacena's anti-de Sitter / conformal field theory correspondence. It posits an exact equivalence between a quantum theory of gravity in a $(d+1)$-dimensional asymptotically anti-de Sitter spacetime and a conformal field theory without gravity on its $d$-dimensional boundary. The equivalence has passed many consistency checks but is not proved, and our own universe is not asymptotically anti-de Sitter.

## Relation to Active Inference
Fields, Friston, Glazebrook and Levin (2022) formulate the free energy principle for generic quantum systems and identify the boundary (Markov blanket) between a system and its environment with a holographic screen: the surface through which all interactions between the two pass, which limits what the system can know. This is an active but contested research programme that borrows the formal idea of boundary encoding and does not rest on the physics of gravity. In ordinary Active Inference the blanket is a statistical construct, and the analogy to black hole entropy bounds is conceptual.

## Related concepts
- [[black_hole_thermodynamics]] — the area law that motivated the principle
- [[quantum_information]] — the information-theoretic language used
- [[entropy]] — the quantity bounded by the boundary area
- [[markov_blanket]] — the statistical boundary compared to a holographic screen
- [[quantum_free_energy]] — the quantum formulation of the free energy principle
- [[quantum_field_theory]] — the framework of the boundary theory

## References
- Bekenstein, J. D. (1973). Black holes and entropy. Physical Review D, 7, 2333-2346.
- Susskind, L. (1995). The world as a hologram. Journal of Mathematical Physics, 36, 6377-6396.
- Bousso, R. (2002). The holographic principle. Reviews of Modern Physics, 74, 825-874.
- Fields, C., Friston, K., Glazebrook, J. F., & Levin, M. (2022). A free energy principle for generic quantum systems. Progress in Biophysics and Molecular Biology, 173, 36-59.
- Maldacena, J. (1998). The large N limit of superconformal field theories and supergravity. Advances in Theoretical and Mathematical Physics, 2, 231-252.
